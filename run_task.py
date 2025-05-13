"""This is the script used for running benchmarks or single tasks.
do_task is the function representing the agent."""

# Benchmark settings are near the end of the file.

# Tip: Use ALT + Z to toggle word wrap in Visual Studio Code
# Tip: Collapsing code (functions, regions, loops, imports,...) makes this project easier to understand.

# Standard Library
from json import load
from time import sleep, time_ns

# 3rd Party
from numpy import array
from PIL import Image, ImageDraw

# Custom scripts
from io_manager import IOManager
from utility import *
from prompts import ActionSelector, ActionSelection, Manager, InitialPlaner, ExplanationAndPlanUpdate, ExplanationAndInitialPlan, AgentMemoryEntry, Plan, Action, ActionType, Memory, InfoPool, Goal, Status, INIT_TIPS, TestConfiguration


def info_log_helper(response_with_meta_data) -> dict:
    """info_log_helper takes the response_with_meta_data tuple and returns a dictionary for easier logging with JSON.

    :param response_with_meta_data: A tuple that contains information about the MLLM call
    :return: A dictionary with where each piece of information is given an appropriate key for naming.
    :rtype: dict
    """

    return {"response": str(response_with_meta_data[0]), "prompt_tokens": response_with_meta_data[1], "model": response_with_meta_data[5], "completion_tokens": response_with_meta_data[2], "cached_tokens": response_with_meta_data[3], "thoughts_tokens": response_with_meta_data[4], "first_successful_call_seconds": response_with_meta_data[6]}


def do_task(inout: IOManager, test_config: TestConfiguration) -> bool:
    """do_task tries to complete the task given in test_config. This function represents the agent.

    :param inout: An object that is used for IO (taking screenshots, creating directories, executing commands ...)
    :type inout: IOManager
    :param test_config: All necessary information for running the task
    :type test_config: TestConfiguration
    :return: False if the execution timed out (max_steps test_config)
        or if the task directory already exists and overwrite is not enabled. True, else.
    :rtype: bool
    """

    # region initialization
    # Some variable initialization
    start_time = time_ns()
    if not inout.manage_task_directory(configuration=test_config):
        return False
    # The InfoPool is used to keep track of important data.
    info_pool = InfoPool()
    # Some general tips are loaded. There is no evolution mechanism in my agent.
    info_pool.tips = INIT_TIPS
    info_pool.instruction = test_config.task
    print("task:", info_pool.instruction)
    print("Logs and images can be found here: " +
          str(inout.current_task_directory))
    print("Tip: In VSC you can just CTRL + click on the path in the terminal to open the directory.")
    # An object generating a prompt for creating the first plan
    initialPlaner = InitialPlaner()
    # An object generating the prompts for creating the following plans, updating the agent memory, and determining the success of the latest action
    manager = Manager()
    # An object for generating the prompts for selecting actions
    operator = ActionSelector()
    print("Model used: ", test_config.model)
    # Logging information is saved here and periodically written to the logging file.
    steps = []
    # Get the size of the Android Screen
    inout.get_size()
    info_pool.width = inout.width
    info_pool.height = inout.height
    info_pool.config = test_config
    # Create first logs
    steps.append({"type": "Configuration"} | test_config.model_dump())
    inout.log_steps(steps)
    # endregion
    # You could call this loop the main loop of the agent.
    for i in range(0, test_config.max_steps):
        steps.append({"step": inout.step, "type": "Start of loop"})
        inout.log_steps(steps)
        # region perception
        screenshot_path_old = inout.screenshot_path
        print("Beginning step number: " + str(i))
        print("Getting data from device.")
        # Get screenshot and UI hierarchy xml file
        inout.get_data("png")
        inout.get_size()
        info_pool.width = inout.width
        info_pool.height = inout.height

        # icon and text detection are still necessary even when using Gemini 2.5 Pro as it has unreliable visual abilities

        # user interface element detection using OmniParser to complement XML data
        # See: https://github.com/microsoft/OmniParser
        # Get a list of bounding boxes. While it would be possible to use another model to describe what exactly was detected, this was not done, as the results are either unreliable or too expensive depending on the used model.

        iou = 0.1
        conf = 0.8
        detected_icon_list = inout.icon_model.predict(
            source=inout.screenshot, conf=0.3, iou=iou, verbose=False)[0].boxes.xyxy.tolist()
        # use easy_ocr to detect and recognize text in the current screenshot
        text_bounding_boxes, recognized_text = inout.recognize_text(text_threshold=conf)

        # remove all detected icons with too much overlap to detected text

        icons = []
        for i in detected_icon_list:
            for j in text_bounding_boxes:
                if(calculate_intersection_over_union(i,j) > iou):
                    break
            else:
                icons.append((toCoord(info_pool,lisToBB(i))))
        text_bounding_boxes = [toCoord(info_pool,i) for i in text_bounding_boxes]
        ui_textual_representation ="Text and icon detection was used to determine the bounding boxes of icons and text in the screenshot. As a high confidence threshold was used, some objects are likely missing. Use the detected elements together with the screenshot and your visual grounding and referring abilities to determine essential coordinates.\nFirst, a list of detected texts:\n"
        lis = []
        for b,t in zip(text_bounding_boxes, recognized_text):
            lis.append(str({"box_2d" : b.flatten().tolist(), "text":t}))
        ui_textual_representation +=str(lis)+"\nSecond, a list of detected icon bounding boxes:\n"
        lis = []
        for b in icons:
            lis.append(str({"box_2d" : b.flatten().tolist()}))
        ui_textual_representation +=str(lis)

        # only get information about the focused element
        if (inout.get_data("xml")):
            root = inout.get_nodes_from_xml()
            focused_list = root.unpack(lis=[], attribute_dict={
                                       "focused": "true"})
            # Either there is a focused element or not
            if len(focused_list) > 0:
                info_pool.focused_element_history.append(focused_list[0])
                info_pool.focused_element_history_character_count.append(
                    len(focused_list[0].attributes["text"]))
            else:
                info_pool.focused_element_history.append(None)
        else:
            # no XML data, means no focused element data and an empty interactable_node_list
            info_pool.focused_element_history.append(None)

        # Check if the keyboard is visible and add to the ui_textual_representation accordingly
        info_pool.keyboard_history.append(inout.is_keyboard_visible())

        prompt = "\nThe virtual keyboard is visible; therefore, text input is possible."
        if (info_pool.keyboard_history[-1]):
            if (info_pool.focused_element_history[-1]):
                prompt += "For text operations or search, the XML data about the current text field is of utmost importance: " + \
                    info_pool.focused_element_history[-1].alternative_str()
                prompt += " This is the currently selected text field you send input to or delete text from."
                prompt += " The text you can edit is: \"" + \
                    info_pool.focused_element_history[-1].attributes["text"]+"\""
                prompt += " If this is not the text field you want to edit, select another text field."
                prompt += " Before doing text operations, you should use this data and the screenshot to determine if you have the right text field selected. Before confirming a search you should make sure that the search text is correct."
            prompt += "The virtual keyboard is visible, which means that you can input text into a text field. Empty text fields usually have some kind of placeholder text in them. Tap on placeholder text to get started. For already filled text fields, tap on the detected text. Placeholder text does not need to be deleted; it usually disappears automatically after input."
        else:
            prompt = "\nNo text input is possible. To input text, tap on a text field to open the virtual keyboard first."
        ui_textual_representation += "\n\n Additional information about possible text input: "+prompt
        info_pool.perception_info_history.append(ui_textual_representation)
        steps.append({"step": inout.step, "type": "Perception",
                     "textual_representation": str(ui_textual_representation), "textual_representation_character_counter": len(str(ui_textual_representation))})
        inout.log_steps(steps)
        # endregion
        # region planning
        if (inout.step == 0):
            # Generating the first plan is harder than updating the plan in later steps.
            # Therefore, generating the first plan gets its own substep, while updating the plan can be combined with other substeps, like reflection or notetaking.
            print("Initial Planning")
            # Creating a custom format for prompts. Gemini and ChatGPT use slightly different inputs. This layer of abstraction simplifies this.
            system_instructions = initialPlaner.init_chat()
            prompt = initialPlaner.get_prompt(info_pool=info_pool)
            planning_message = [system_instructions,
                                ("user", [inout.screenshot_path, prompt])]
            # Get the response from the MLLM
            response_with_meta_data = inout.query(simplified_messages=planning_message, task_description=(
                info_pool.instruction + str(inout.step)), output_format_json=ExplanationAndInitialPlan, test_config=test_config)
            response = response_with_meta_data[0]
            # Initialize the plan and agent memory
            info_pool.plan = Plan(**(response["InitialPlan"]))
            info_pool.memory = Memory()
            info_pool.memory.memory = [AgentMemoryEntry(
                **i) for i in response["ListOfDesiredInformation"]]
            steps.append({"step": inout.step, "type": "Initial Planning", "prompt": str(
                planning_message)} | info_log_helper(response_with_meta_data))
            inout.log_steps(steps)
        else:
            print("Manager")
            # Prompt
            system_instructions = manager.init_chat()
            prompt = manager.get_prompt(info_pool=info_pool)
            if(test_config.action_vis):
                planning_message = [system_instructions, ("user", [screenshot_path_vis, inout.screenshot_path,
                        prompt])]
            else:
                planning_message = [system_instructions, ("user", [screenshot_path_old, inout.screenshot_path,
                        prompt])]
            # Get response
            response_with_meta_data = inout.query(simplified_messages=planning_message, task_description=(
                info_pool.instruction + str(inout.step)), output_format_json=ExplanationAndPlanUpdate, test_config=test_config)
            response = response_with_meta_data[0]
            # Update active Subgoal
            updated_subgoal = Goal(**(response["activeSubgoalUpdate"]))
            info_pool.outcome_description_history.append(
                response["ActionOutcomeDescription"])
            for i in range(0, len(info_pool.plan.Subgoals)):
                if info_pool.plan.Subgoals[i].ID == updated_subgoal.ID:
                    info_pool.plan.Subgoals[i] = updated_subgoal
                    break
            # Select next Subgoal
            info_pool.plan.activeSubgoalId = response["nextSubgoalId"]
            # Update agent memory
            new_entries = [AgentMemoryEntry(
                **i) for i in response["ListOfUpdatedAndAdditionallyDesiredInformation"]]
            for i in range(0, len(info_pool.memory.memory)):
                for j in new_entries:
                    if j.VariableName == info_pool.memory.memory[i].VariableName:
                        info_pool.memory.memory[i] = j
                        break
            for i in new_entries:
                for j in info_pool.memory.memory:
                    if j.VariableName == i.VariableName:
                        break
                else:
                    info_pool.memory.memory.append(i)
            steps.append({"step": inout.step, "type": "Planning, Reflection, and Notetaking", "prompt": str(
                planning_message)} | info_log_helper(response_with_meta_data))
            inout.log_steps(steps)
            # Stop execution if all Subgoals are completed. Sometimes, Gemini Flash 2.0 forgets to complete a subgoal before selecting the next. So this will also stop if there is no valid active subgoal.
            br = info_pool.plan.activeSubgoalId in [
                j.ID for j in info_pool.plan.Subgoals]
            for i in info_pool.plan.Subgoals:
                if i.status != Status.COMPLETED and br:
                    break
            else:
                steps.append({"type": "final", "reported_result": "Finished", "task_seconds": (
                    time_ns() - start_time) / (1e9)})
                inout.log_steps(steps)
                print("Agent finished!")
                return True
        # endregion
        # region action_selection
        print("Action Selection")
        # Prompt
        system_instructions = operator.init_chat()
        prompt = operator.get_prompt(info_pool)
        planning_message = [system_instructions, ("user", [inout.screenshot_path,
                prompt])]
        # Get response
        response_with_meta_data = inout.query(simplified_messages=planning_message, task_description=(
            info_pool.instruction + str(inout.step)), output_format_json=ActionSelection, test_config=test_config)
        response = response_with_meta_data[0]
        # The expected behavior of the action is saved for the next reflection
        info_pool.summary_history.append(response["ActionDescription"])
        steps.append({"step": inout.step, "type": "Action Selection", "prompt": str(
            planning_message)} | info_log_helper(response_with_meta_data))
        inout.log_steps(steps)
        # endregion
        # region action execution
        # The selected action is executed
        action = Action(**response["action"])
        info_pool.action_history.append(action)
        command_list = action.get_action_command(info_pool)
        for j in command_list:
            inout.execute_adb_command(j)
        if action.name == ActionType.wait:
            print("Waiting for 10 seconds.")
            sleep(6)  # 6 + 4 = 10
        if action.name == ActionType.answer:
            inout.answer = action.text
            steps.append({"type": "final", "reported_result": "Finished", "task_seconds": (
                    time_ns() - start_time) / (1e9)})
            inout.log_steps(steps)
            print("Agent finished!")
            return True

        # The action is visualized in the ui_number_box.jpg, if this helps for reflection is questionable, but it makes interpreting the images easier for a human.
        # This draws a red circle around the tap or long_press position:
        x0,y0,x1,y1 = action.get_coordinates(info_pool)
        img = Image.open(inout.screenshot_path)
        screenshot_path_vis = str(inout.screenshot_path)[:-4]+"_vis.jpg"
        box_path = str(inout.screenshot_path)[:-4]+"_box.jpg"
        img_drw = ImageDraw.Draw(img)
        if action.name == ActionType.tap or action.name == ActionType.long_press:
            img_drw.circle((x0, y0), outline="red", width=4, radius=20)
            # this marks the swipe with a red arrow
        if action.name == ActionType.swipe:
            draw_arrow(img, array([x0, y0]), array(
                [x1, y1]), width = 4)
        img.save(screenshot_path_vis, quality=100)
        for i in icons:
            bb = fromCoord(info_pool, i).flatten()
            bb = (tuple(bb[0:2]),tuple(bb[2:]))
            img_drw.rectangle(bb, outline="red", width=2)
        for i in text_bounding_boxes:
            bb = fromCoord(info_pool, i).flatten()
            bb = (tuple(bb[0:2]),tuple(bb[2:]))
            img_drw.rectangle(bb, outline="red", width=2)
        img.save(box_path, quality=100)

        steps[-1]["action"] = action.model_dump_json()
        print(action.model_dump_json())
        # endregion

        # The program sleeps for 4 seconds so that the consequences of the last action can happen.
        # In AndroidWorld, the detection of a stable UI seems to be handled with more sophistication.
        sleep_seconds = 4
        print("Sleeping for "+str(sleep_seconds))
        sleep(sleep_seconds)
        inout.step = inout.step + 1

    # If this loop is exited, the agent did not manage to achieve its goal in the required number of steps.
    steps.append({"type": "final", "reported_result": "Aborted",
                 "task_seconds": (time_ns() - start_time) / (1e9)})
    steps.append({"actual_success": False})
    inout.log_steps(steps)
    return False


if __name__ == "__main__":
    # This is usefull, if you want to call the script from a different folder.
    from pathlib import Path
    p = Path(__file__).resolve().parent
    # Selecting the folder where the test data should be saved to.
    folder = "task"
    inout = IOManager(base_directory=p/folder, verbose=False,
                      overwrite=False, wait_between_calls=False)
    print("Running tasks:")
    with open(p/"benchmark.json", "r") as file:
        tasks = load(file)["tasks"]

    # Run the benchmark:
    
    # You can run this script to run the benchmark, but the device state has to be set by hand between tasks.
    # Set the device state to the state it would have for a successful task and delete the history for Chrome,
    # the search history for Maps.

    test_configs = []
    model = "gemini-2.5-pro-preview-03-25"
    # model = "gemini-2.0-flash"
    sub_folder = "gemini-2.5-pro-preview-03-25"
    test_configs.append(TestConfiguration(task="", model=model, sub_folder=sub_folder, action_vis= True))

    for test_config in test_configs:
        for i in tasks[:]:
            # If the task is not skipped or times out, annotation for success is handled here.
            test_config.task = i["instruction"]
            test_config.max_steps = (len(i["human_reference_operations"]) * 4) if test_config.model in ["gemini-2.0-flash"] else (len(i["human_reference_operations"]) * 2)
            inout.base_directory = (p/folder)/test_config.sub_folder
            # audio feedback
            t3 = time_ns()
            play_beep()
            if (do_task(inout, test_config=test_config)):
                print("Current task finished. Logs and images can be found here: " +
                      str(inout.current_task_directory))
                play_beep()
                play_beep()
                play_beep()
                result = 2
                while (result != 1 and result != 0):
                    try:
                        result = int(
                            input("Was the task successful? Input 1 for success, 0 for failure:"))
                    except Exception as e:
                        pass
                inout.estimate_cost(test_config.model)
                with open(inout.current_task_directory/"steps.json") as f:
                    li = load(f)
                    li.append({"actual_success": bool(result)})
                inout.log_steps(li)
            if(time_ns() - t3 > 4000000000):
                input("Please reset the device for the next test.")
