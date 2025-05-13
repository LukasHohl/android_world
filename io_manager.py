"""In this file, a class is defined that handles most IO operations for the agent."""

# Standard Library
from datetime import datetime, timezone
from os import environ, remove
from json import dump, loads, JSONDecodeError
from sys import exit
from pathlib import Path, PurePosixPath
import subprocess
from shutil import rmtree
from time import sleep, time_ns
from xml.etree.ElementTree import iterparse

# 3rd Party
from google import genai
from google.genai import types
from numpy import array, asarray
from pandas import read_csv, concat, DataFrame
from PIL import Image
import easyocr
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import torch

# Custom scripts
from ui_node_xml import UINode

class IOManager:
    """The IOManager does input-output related operations like: loading settings, initializing folders, selecting, controlling the adb device and sending requests to MLLMs"""

    def __init__(self, base_directory, settings_row_number=0, lang=['de', 'en'], verbose=True, overwrite=False, wait_between_calls=True):
        """Settings are loaded, directories are created, and the adb device is selected.

        :param base_directory: The directory where, for each task, a new directory is created.
        :type base_directory: str
        :param settings_row_number: A relic of the previous agent, most settings were moved to TestConfiguration. Setting this to 0 is probably correct.
        :type settings_row_number: int
        :param lang: The list of languages EasyOCR should detect. Chinese, English, German, etc. While EasyOCR is not the most accurate, it provides many options.
        :type lang: list[str]
        :param verbose: Set this to true if you want to print all adb command results to the terminal.
        :type verbose: bool
        :param overwrite: Set this to true if you want tasks with identical parameters to overwrite previous runs.
        :type overwrite: bool
        :param wait_between_calls: Should the agent wait a minimum amount of time between MLLM calls?
        :type wait_between_calls: bool
        """
        self.answer = ""
        # The number of steps the agent is in. This is necessary for naming files.
        self.step = 0
        # A dictionary saving when each model was called last; this is used to not call Gemini too often because of usage limits for the free API key.
        self.last_call_time_per_model = dict()
        # The current Screenshot
        self.screenshot = None  # RGB Image
        # The path to the current Screenshot on disk
        self.screenshot_path = None
        # The path to the XML
        self.xml_path = None
        self.cuda_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.overwrite = overwrite
        self.wait_between_calls = wait_between_calls
        # The last time a log was created is saved. This way, the time between steps can be easily extracted.
        self.last_log_time = None
        # Downloading OmniParser model, if necessary. This might take a while.
        model_path = hf_hub_download(
            repo_id="microsoft/OmniParser-v2.0", filename="icon_detect/model.pt", repo_type="model")
        # Load the icon detection and text recognition model
        self.icon_model = YOLO(model_path).to(self.cuda_device)
        self.easy_reader = easyocr.Reader(lang)
        # The adb device
        self.device = None
        # The settings are where you can specify the name of the api key variables.
        self.settings = {}
        # Setting paths
        self.base_directory = Path(base_directory)
        self.current_task_directory = None
        self.script_directory_path = Path(__file__).resolve().parent
        self.log_path = self.script_directory_path/"token_usage.csv"
        # Load the settings
        try:
            for key, value in read_csv(self.script_directory_path/"settings.csv").loc[settings_row_number, :].items():
                self.settings[key] = value
        except Exception as e:
            print(str(e) + " Could not load settings! ")
            exit(1)

        # Select the Android device
        while True:
            adb_device_list = [x.split("\t")[0] for x in self.execute_adb_command(
                "devices").splitlines()[1:]]
            for i, value in enumerate(adb_device_list):
                print(str(i) + ": " + str(value))
            length = len(adb_device_list)
            if length == 0:
                print("Could not find any adb devices! Connect a device and try again!")
                exit(1)
            elif length == 1:
                print(
                    "Device: " + str(adb_device_list[0]) + " was automatically selected, because it was the only available device.")
                self.device = adb_device_list[0]
                break
            else:
                try:
                    n = int(input(
                        "Select a device by number! The number should be an integer between 0 and "+str(len(adb_device_list)-1)+":\n"))
                    if n >= 0 and n < length:
                        print("You selected: " + str(adb_device_list[n]))
                        self.device = adb_device_list[n]
                        break
                except Exception as e:
                    print("Try Again!")
        # get the screen size
        self.get_size()
        self.android_directory = PurePosixPath(
            self.settings["ANDROID_DIRECTORY"])
        self.execute_adb_command("shell mkdir " + str(self.android_directory))

    def get_size(self):
        """get_size gets the screen dimensions from the adb device."""

        i = self.execute_adb_command("shell wm size")
        if i:
            self.width, self.height = tuple(
                [int(x) for x in i.split(": ")[-1].split("x")])
        else:
            print("Failed to get size of adb device: " + self.device)
            exit(1)

    def get_data(self, type):
        """get_data either gets a Screenshot, or an XML describing the UI from the adb device.

        :param type: png to get a Screenshot, XML to get the uidump
        :type type: str
        """
        """This function is used to get XML and Screenshot data, legal types are png and XML"""
        s = self.android_directory/("file."+type)
        # Delete the file on the Android device if it already exists.
        self.execute_adb_command(f"shell rm {s}")
        if type == "png":
            i = self.execute_adb_command(f"shell screencap -p {s}")
            file_name = ("screenshot_"+str(self.step)+".png")
        elif type == "xml":
            i = self.execute_adb_command(
                f"shell uiautomator dump --compressed --window ALL {s}")
            file_name = ("ui_"+str(self.step)+".xml")
        else:
            print(type+ " - Error! File Type not supported!")
            exit(1)
        if i:
            sleep(0.1)
            target_path = self.current_task_directory/(file_name)
            i = self.execute_adb_command(f"pull {s} " + str(target_path))
            if i:
                if type == "png":
                    self.screenshot = Image.open(target_path).convert("RGB")
                    remove(target_path)
                    file_name = ("ui_"+str(self.step)+".jpg")
                    target_path = self.current_task_directory/file_name
                    self.screenshot.save(target_path, quality=100)
                    self.screenshot_path = target_path
                elif type == "xml":
                    self.xml_path = target_path
                else:
                    pass
                # Delete file from device storage after transfer.
                self.execute_adb_command(f"shell rm {s}")
                return i
        print("Failed to get data")
        return None

    def execute_adb_command(self, shell_command: str):
        """execute_adb_command executes an adb shell command.

        :param shell_command: The shell command that should be executed on the Android device. No 'adb' prefix is necessary.
        :type shell_command: str
        :return: This function returns either None, True or the stdout.
        """
        shell_command = "adb " + \
            ("" if self.device is None else ("-s "+self.device + " ")) + shell_command
        i = subprocess.run(shell_command, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (self.verbose):
            print("shell command:", shell_command, "Result:", i)
        if i.returncode == 0:
            s = i.stdout.strip()
            if s == "":
                return True
            else:
                return s
        return None

    def manage_task_directory(self, configuration) -> bool:
        """manage_task_directory creates a directory for a configuration based on its hash value. Existing folders are overwritten if self.overwrite == True

        :param configuration: A TestConfiguration object that is hashed to generate the directory name.
        :type configuration: TestConfiguration
        :return: This returns if a new directory was created.
        :rtype: bool"""
        # Given a new task, the step number is reset. This is important for filenames.
        self.step = 0
        if (not self.base_directory.exists()):
            raise FileNotFoundError(
                "The base directory for the current task does not exist: "+str(self.base_directory))
        try:
            folder = configuration.generate_hash()
            self.current_task_directory = self.base_directory / folder
            # Only delete existing directories if requested to avoid deleting expensive test results.
            if (self.current_task_directory.exists()):
                if (self.overwrite):
                    print("Overwriting existing data!")
                    rmtree(self.current_task_directory)
                    self.current_task_directory.mkdir()
                    return True
                print("Data for this task configuration already exists! Skip.")
                return False
            else:
                self.current_task_directory.mkdir()
                return True
        except Exception as e:
            print(str(e)+": Failed to create directory: " +
                  str(self.current_task_directory))
        print("Something went wrong!")
        self.current_task_directory = None
        return False

    def delay(self, model_name):
        pause = 1
        # You can control the pauses per model here
        if model_name in ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-2.0-flash","gemini-2.5-flash-preview-04-17"]:
            pause = 4
        elif model_name in ["gemini-2.5-pro-preview-03-25","gemini-1.5-pro", "gemini-2.0-pro-exp", "gemini-2.0-pro-exp-02-05", "gemini-2.5-pro-exp-03-25", "gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-preview-05-06"]:
            pause = 30
        else:
            raise Exception("Failure! Model: " +
                            str(model_name) + " is not supported!")
        # Free Gemini keys only have a limited number of allowed calls per minute.
        if (self.wait_between_calls and (model_name in self.last_call_time_per_model)):
            current_time = time_ns()
            last_call_time = self.last_call_time_per_model[model_name]
            if (current_time - last_call_time < 1_000_000_000 * pause):
                time_to_sleep = (1_000_000_000 * pause -
                                 (current_time - last_call_time))/1_000_000_000.0
                print("The last api call is too close, sleeping for: " +
                      str(time_to_sleep))
                sleep(time_to_sleep)
    def query(self, task_description, simplified_messages, output_format_json, test_config):
        """This function is used to handle API calls to Gemini or ChatGPT.

        :param task_description: A string used for logging token usage. Does not influence agent execution.
        :type task_description: str
        :param simplified_messages: A list of tuples (role, text) and images. Used to simplify prompt generation for ChatGPT and Gemini
        :param output_format_json: The BaseModel that represents the exact JSON format that should be returned.
        :type output_format_json: BaseModel
        :param test_config: The TestConfiguration object contains information like the temperature or the model that should be used.
        :type test_config: TestConfiguration
        :return: A tuple containing a dictionary that can be converted to the BaseModel output format and other information like token usage.
        :rtype: tuple
        """
        model_name = test_config.model
        print("Making a call to: " + model_name)
        self.delay(model_name)
        t = test_config.temperature
        top_p = test_config.top_p
        for i in range(4):
            # TODO Everything seems to work, but more error handling could be added.
            try:
                # the time for an api call is measured, the waiting time is not interesting and can be calculated
                current_time = time_ns()
                if model_name in ["gemini-2.5-pro-preview-03-25","gemini-2.5-pro-exp-03-25","gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-pro-exp", "gemini-2.0-pro-exp-02-05", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06"]:
                    result = self.query_gemini(
                        task_description, simplified_messages, output_format_json, test_config)
                else:
                    raise Exception("Failure! Model: " +
                            str(model_name) + " is not supported!")
                self.last_call_time_per_model[model_name] = time_ns()
                test_config.temperature = t
                test_config.top_p = top_p
                return result + ((time_ns()-current_time)/1e9,)
            except JSONDecodeError as e:
                # JSONDecodeError happened when Gemini returned too many tokens and the JSON was incomplete as the maximum number of tokens was reached.
                # because for some reasons there are JSONDecodeErrors and the need
                test_config.temperature = min(t + 0.1, 1.0)
                test_config.top_p = min(t + 0.1, 0.9)
                print("Modifying temperature and top_p: ")
                print("This could mean that the maximum number of completion tokens needs to be increased.")
                print(type(e).__name__)
                time_to_sleep = 0
                print("Sleeping for: " + str(time_to_sleep+1))
                sleep(time_to_sleep + 1)
            except genai.errors.ServerError as e:
                print(type(e).__name__)
                time_to_sleep = 59
                print("Sleeping for: " + str(time_to_sleep+1))
                sleep(time_to_sleep + 1)
        raise Exception("Too many errors!")

    def query_gemini(self, task_description, simplified_messages, output_format_json, test_config):
        """This function is used to handle calls to Gemini.

        :param task_description: A string used for logging token usage. Does not influence agent execution.
        :type task_description: str
        :param simplified_messages: A list of tuples (role, text) and images. Used to simplify prompt generation for ChatGPT and Gemini
        :param output_format_json: The BaseModel that represents the exact JSON format that should be returned.
        :type output_format_json: BaseModel
        :param test_config: The TestConfiguration object contains information like the temperature or the model that should be used.
        :type test_config: TestConfiguration
        :return: A tuple containing a dictionary that can be converted to the BaseModel output format and other information like token usage.
        :rtype: tuple
        """
        client = genai.Client(
            api_key=environ[self.settings["GEMINI_ENVIRONMENT_VARIABLE"]])
        temperature = test_config.temperature
        model_name = test_config.model
        messages = []
        system_instructions = []
        for x, y in simplified_messages:
            if (x == "model"):
                system_instructions += y
                continue
            j = []
            for i in y:
                if str(i).endswith(".jpg"):
                    j.append(Image.open(i))
                else:
                    j.append(i)
            messages.append(j)
            generation_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=test_config.top_p,
            max_output_tokens=test_config.max_tokens,
            response_mime_type="application/json",
            response_schema= output_format_json,
            system_instruction=system_instructions,
            thinking_config= types.ThinkingConfig(thinking_budget=test_config.max_thinking_tokens)
        )
        response = client.models.generate_content(
            model=model_name, contents=messages, config=generation_config)
        model_name = response.model_version
        self.log(task_description, response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count,
                (response.usage_metadata.cached_content_token_count if response.usage_metadata.cached_content_token_count else 0),(response.usage_metadata.thoughts_token_count if response.usage_metadata.thoughts_token_count else 0), model_name)
        text = parse_json(response.text)
        return loads(text), response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count, (response.usage_metadata.cached_content_token_count if response.usage_metadata.cached_content_token_count else 0),(response.usage_metadata.thoughts_token_count if response.usage_metadata.thoughts_token_count else 0), model_name

    def log(self, task_description: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int, thoughts_tokens : int, model: str):
        """log is used to log the token usage.

        :param task_description: A string used for logging token usage. Does not influence agent execution.
        :type task_description: str
        :param prompt_tokens: The number of tokens that were used to represent the input.
        :type prompt_tokens: int
        :param completion_tokens: The number of tokens that were used for the response.
        :type completion_tokens: int
        :param cached_tokens: The number of cached tokens.
        :type cached_tokens: int
        :param model: The name of the MLLM that was used
        :param model: str
        """
        """This function logs the token usage"""
        path = self.log_path
        new_data = {'prompt_tokens': [prompt_tokens],
                    'completion_tokens': [completion_tokens],
                    'thoughts_tokens' : [thoughts_tokens],
                    'cached_tokens': [cached_tokens],
                    'description': [task_description],
                    'model': [model]}
        try:
            df = read_csv(path)
            concat([df, DataFrame(data=new_data)], ignore_index=True,
                   axis=0).to_csv(path, index=False)
        except FileNotFoundError:
            DataFrame(data=new_data).to_csv(path, index=False)
    def estimate_cost(self, name):
        try:
            df = read_csv(self.log_path)
            df = df.groupby(["model"]).sum(numeric_only=True)
            for  m, r in df.iterrows():
                pt = r['prompt_tokens']
                ct = r['completion_tokens'] + r['thoughts_tokens']
            print(m,pt, ct, (pt*1.25+ct*10)/1e6)
        except FileNotFoundError:
            print("Could not estimate cost. No file was found.")

    def log_steps(self, steps):
        """log_steps opens the logging data and saves the current list of dictionaries containing information about the current run.

        :param steps: List of dictionaries containing all logging data for the run.
        :type steps: list[dict]
        """
        if (not self.last_log_time):
            self.last_log_time = time_ns()
        with open(self.current_task_directory/"steps.json", "w") as f:
            steps[-1]["image_width"] = self.width
            steps[-1]["image_height"] = self.height
            date = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            steps[-1]["date"] = date
            steps[-1]["time since last log:"] = (
                time_ns() - self.last_log_time)/1e9
            dump(steps, f, indent=4)
            self.last_log_time = time_ns()

    def is_keyboard_visible(self) -> bool:
        """is_keyboard_visible returns if the virtual keyboard is visible.

        :return: Is the virtual keyboard visible?
        :rtype: bool
        """
        shell_command = "adb " + \
            ("" if self.device is None else ("-s "+self.device + " ")) + \
            " shell dumpsys input_method"
        r = subprocess.run(shell_command, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in r.stdout.splitlines():
            if "mInputShown=true" in line:
                return True
        else:
            return False

    def recognize_text(self, text_threshold=0.80, y_threshold=0.2, paragraph=True):
        """recognize_text uses EasyOCR to recognize text and returns a list of bounding boxes and a list of texts of the same length.

        :param text_threshold: The text confidence threshold.
        :type text_threshold: float
        :param y_threshold: The y_threshold controls which text are merged into the same paragraph if they are close horizontally.
        :type y_threshold: float
        :return: A tuple containing a bounding box and a list of corresponding text.
        :rtype: tuple
        """
        # easy ocr documentation: https://www.jaided.ai/easyocr/documentation/
        result = self.easy_reader.readtext(asarray(
            self.screenshot), paragraph=paragraph, text_threshold=text_threshold, y_ths=y_threshold)
        bounding_box_list = [array([i[0][0], i[0][2]]) for i in result]
        text_list = [i[1] for i in result]
        return bounding_box_list, text_list
    

    def get_nodes_from_xml(self):
        """This function parses the XML data and returns the root node.

        :return: The root node of a tree of UINode representing the user interface.
        :rtype: UINode"""
        self.get_size()
        root = None
        list_of_nodes_from_root = []
        for event, elem in iterparse(self.xml_path, events=["start", "end"]):
            if event == "start":
                if elem.tag != "node":
                    if elem.tag == "hierarchy":
                        node = UINode(attributes=elem.attrib, bounding_box=array(
                            [[0, 0]]), root=True)
                        root = node
                    else:
                        print("Unexpected element: "+elem.tag)
                else:
                    boundingbox = array([array([int(i) for i in vector.split(
                        ",")]) for vector in elem.attrib["bounds"][1:-1].split("][")])
                    node = UINode(attributes=elem.attrib,
                                  bounding_box=boundingbox)
                    if len(list_of_nodes_from_root) > 0:
                        list_of_nodes_from_root[-1].children.append(node)
                list_of_nodes_from_root.append(node)
            elif event == "end":
                list_of_nodes_from_root.pop()
        if (root == None):
            print("The program expects a root node, but there is none.")
            exit(1)
        return root

# Source: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb
# 2.5 exp does not seem to be able to do structured output yet without error?
def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
