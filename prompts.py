"""This file contains code to generate prompts for MLLM calls.
I am using heavily modified code/prompts from Mobile-Agent-E here:
Source: https://github.com/X-PLUG/MobileAgent
Using Basemodel for prompting heavily modified the code.
Parts of the prompts were from AppAgent: https://github.com/TencentQQGYLab/AppAgent
Presumably, the AppAgent prompts were entirely deleted, but it is hard to keep track.
"""

# Standard Library
from abc import ABC, abstractmethod
from base64 import b64encode
from enum import Enum
from hashlib import sha256

# 3rd Party
from pydantic import BaseModel, Field
from typing import List, Annotated

# region string_definitions
# Tips are necessary to help the agent overcome misconceptions.
detailed_step_by_step_explanation = "Provide a detailed step by step explanation of your rationale for "
INIT_TIPS = """0. Do not add any payment information. If you are asked to sign in, ignore it or sign in as a guest if possible. Close any pop-up windows when opening an app. Pop-ups can usually be closed by either tapping a special button on the pop-up or, if such a thing does not exist, by tapping somewhere else on the screen.
1. Make sure that the search field really contains the correct search text before starting a search. Text below the search field might only be a suggestion.
2. Screenshots may show partial text in text boxes from your previous input; this does not count as an error.
3. When creating new Notes, you do not need to enter a title unless the user specifically requests it.
4. Often, after entering an email address or other information into some text field, you need to hit enter to confirm the input. It is possible that without confirmation, a list of suggestions might appear for some text fields.
5. UI elements like a switched toggle or other colorful indicators hint at the state of the program.
6. In order to enable or disable some options, prioritize clicking on buttons or sliders. Only if there is no such thing as an option, try tapping on text.
7. Do not just start inputting text. You first have to be sure you selected the right text field. Placeholder text can be great for finding the correct text field.
8. If you want to change some settings for the Android device, you usually have to open the 'Settings' app.
9. If you do not see an app on the home screen, check the app drawer to see if you can find it there.
10. Opening the app drawer by swiping up requires that the swipe does not start too close to the bottom of the screen. Starting the upward swipe at y = 900 works to open the app drawer if the swipe is long enough.
11. When applicable, interact with the center of UI elements, e.g., start swipes, tap, and long_presses on the center of the UI element you want to interact with.
12. Do not input or delete text character by character if possible. Prefer the 'text' and 'delete' actions for efficient text operations.
13. To interact more often with the intended target for 'tap', 'long_press', or 'swipe', try to interact with the center of the biggest icon or text, which interaction leads closer to the fulfillment of the subgoal. For text fields, prefer the center of the existing text. To tap or long_press a button, interact with the center of the button. For swiping sliders, select points on the central axis in the direction of movement.
14. If other user interface elements mostly cover an element you want to interact with, try to use a swipe to uncover it first.
15. You are really bad at recognizing sliders. If you see a search bar, when you need a slider try swiping on it to check if it is actually a slider. Important: Do not just swipe on the screen without an element that could be a slider. Seriously, do not just assume that a slider is somewhere without a good reason like a long object that you think could be searchbar.
16. If you want to set a slider all the way to the right or the left, swipe more than needed to ensure the correct result.
17. Pressing 'Back' while the virtual keyboard is open will close the keyboard. If you want to navigate to the previous page, prefer tapping on user interface elements to save a step.
18. Some tasks ask you to work with apps like 'Simple Gallery Pro', 'Simple Calendar Pro', or other 'Simple Something Pro' apps. The name of these Simple Pro apps is displayed differently: e.g., for 'Simple Gallery Pro' the app name is 'Gallery', for 'Simple Calendar Pro' it is 'Calendar' and so on.
19. Instead of 'Simple SMS Messenger' use the 'Simple SMS Messenger' app.
"""

# Text, that is used multiple times:
helpful_android_assistant = "You are part of a helpful AI agent for operating Android mobile phones. You aim to fulfill the user's instruction through thorough, step-by-step reasoning."
action_space_string = "To fulfill the user instruction, you can only plan using the following actions: tap, long_press, swipe, input text into the currently selected text field, delete text from the currently selected text field, wait for 10 seconds, and you can also raise keyevents via 'adb shell input keyevent', e.g., press the back, home, enter buttons or similar. Finally, you can send an answer to the user. Only use answer as the last action, as it terminates the execution."
orientation_information = "Orientation information: The top left corner of the screen is the origin. The x and y axis go horizontally and vertically, respectively. Coordinate values are normalized to 0-1000. Point are given in [y, x] format, Bounding box in [y_min, x_min, y_max, x_max] format. \nTips for Swiping: For example, swiping from position [500, 500] to position [250, 500] will swipe upwards to review additional content below. Swiping from [250, 500] to position [500, 500] would be Swiping down. Swiping from [500, 250] to position [500, 500] would be Swiping to the right. Swiping from [500, 500] to position [500, 250] would be Swiping to the left."
# endregion

# region JSONFormat
# Classes inheriting from BaseModel can be easily converted to a JSON schema.
# These schemas supported by multiple Gemini and ChatGPT versions are a great way to (mostly) guarantee the output format.

class ActionType(Enum):
    """The type/category of an action."""
    tap = "Tap on the screen at start_point. This is often used for primary interaction with most user interface elements."
    long_press = "Long press on the screen at start_point. For some user interface elements, this is used for secondary interactions like selecting and elements for deletion."
    swipe = "Swipe on the screen from start_point to end_point."
    text = "Append the provided text to the end of the text in the currently focused text field."
    delete = "Deletes all text in the currently focused text field. If there is no information, this action attempts to delete 100 characters."
    send_keyevent = "Use 'adb shell input keyevent keycode' to raise a keyevent."
    wait = "Makes the agent wait for 10 seconds. Use this if a page is not loaded."
    answer = "If the task requires it, use this to send an answer to the user. Only use answer as the last action, as it terminates the execution."

Point_2d = Annotated[List[int], Field(min_length=2, max_length=2, description = "Point in [y, x] format.")]
Box_2D = Annotated[List[int], Field(min_length=4, max_length=4, description = "Bounding box in [y_min, x_min, y_max, x_max] format.")]
class Action(BaseModel):
    """Action represents an action the agent can execute."""
    name: ActionType
    start_point : Point_2d= Field(description="The Point on screen that should be interacted with. Set this for tap, long_press and swipe. Set start_point coordinates to 0 for all other ActionType.")
    end_point : Point_2d= Field(description="The Point on screen that should be swiped to. Set this for swipe. Set end_point coordinates to 0 for all other ActionType.")
    text: str = Field(
        description="The text input. Set this for text or to return an answer. For an answer the text is returned unmodified. For text input the text is divided by each appearing '\\n' in a series of text inputs and pressing the enter button. This can be used to start a new line or confirm text input in search fields. For example, text = '\nabc\n' will result in a press of the enter button, followed by writing 'abc', followed by another press of the enter button. To save an additional step, you can add '\\n' to the end of a text you want to input into a search field to start the search. Set text to \"\" for all other ActionType.")
    keycode: str = Field(description= "If the action is of the type send_keyevent, this is the keycode that will be send. Examples of important keycodes are: 'KEYCODE_BACK' to press the back button, '66' to press the enter button, and '3' to press the home button. Set keycode to '' for all other action types.")
    def get_coordinates(self, info_pool):
        x0 = self.start_point[1]/1000.0*info_pool.width
        y0 = self.start_point[0]/1000.0*info_pool.height
        x1 = self.end_point[1]/1000.0*info_pool.width
        y1 = self.end_point[0]/1000.0*info_pool.height
        return x0,y0,x1,y1
    def get_action_command(self, info_pool) -> list[str]:
        """get_action_command converts an Action to list of shell commands.

        :param info_pool: The info_pool contains important information.
        :type info_pool: InfoPool
        :return: A list of shell commands.
        :rtype: list[str]
        """
        s = []
        duration = 1000
        x0,y0,x1,y1 = self.get_coordinates(info_pool)
        if self.name in [ActionType.wait, ActionType.answer]:
            pass
        elif self.name == ActionType.send_keyevent:
            s.append("shell input keyevent "+self.keycode)
        elif self.name == ActionType.tap:
            s.append("shell input tap " + str(x0) +
                     " " + str(y0))
        elif self.name == ActionType.text:
            s.append("shell input keyevent KEYCODE_MOVE_END")
            text = self.text
            text = text.replace("\\n", "\n")
            i = 0
            while (i < len(text)):
                next_text = ""
                while i < len(text) and text[i] != "\n":
                    next_text += text[i]
                    i = i+1
                else:
                    if (len(next_text) > 0):
                        # See: https://github.com/senzhk/ADBKeyBoard
                        s.append(
                            f"shell am broadcast -a ADB_INPUT_B64 --es msg {b64encode(next_text.encode('utf-8')).decode('ascii')}")
                        continue
                if text[i] == "\n":
                    s.append(" shell input keyevent 66")
                    i = i+1
                    continue
        elif self.name == ActionType.long_press:
            s.append(
                f"shell input swipe {x0} {y0} {x0} {y0} {duration}")
        elif self.name == ActionType.swipe:
            s.append(
                f"shell input swipe {x0} {y0} {x1} {y1} {duration}")
        elif self.name == ActionType.delete:
            s.append("shell input keyevent KEYCODE_MOVE_END")
            repeat = 100
            if (info_pool.keyboard_history[-1]):
                repeat = info_pool.focused_element_history_character_count[-1]
            for i in range(repeat):
                s.append("shell input keyevent KEYCODE_DEL")
        else:
            raise NotImplementedError(
                f"The action with the name: {self.name} is not implemented.")
        return s


class InformationCompleteness(Enum):
    "The status of a piece of information in the Agent Memory."
    COMPLETE = "A complete piece of information."
    WANTED = "This piece of information is sought. No content or partial information is currently available."


class AgentMemoryEntry(BaseModel):
    """Only create Agent Memory Entries for pieces of information essential to fulfill the Subgoal and the user instruction. Do not create Agent Memory Entries for general knowledge that you already know. Only create Agent Memory Entries for pieces of information you need to retrieve."""
    VariableName: str = Field(
        description="Use a descriptive string to describe what information the AgentMemory entry should contain.")
    Content: str = Field(
        description="The information the MemoryEntry contains. This can be an empty or incomplete string if the information still has to be retrieved.")
    Status: InformationCompleteness = Field(
        description="The status describes whether the information gathering for the AgentMemoryEntry is completed or if there is still information that has to be collected for the AgentMemoryEntry.")


class Status(Enum):
    "The Status of a Goal."
    NO_PROGRESS_YET = "no progress yet"
    IN_PROGRESS = "in progress"
    COMPLETED = "completed"


class Goal(BaseModel):
    """The class Goal is used to break down the user instruction into concrete, manageable parts."""
    ID: int = Field(description="The unique ID of the Goal.")
    Title: str = Field(description="A descriptive title.")
    CompletionCriteria: str = Field(
        description="A string describing when the Goal is considered completed.")
    ProgressDescription: str = Field(description="A string describing the Progress and the history for completing the Goal. In order to know when the Goal is completed, it is very important to keep a detailed record of what you have already done. When creating a new ProgressDescription, process the information from the old ProgressDescription. Mention what approaches and actions did not work so that you can learn from mistakes.")
    PlannedNextSteps: list[str] = Field(
        description="A Goal has PlannedNextSteps, a list of strings; this should be a step-by-step plan of the next steps necessary to come closer to the completion of the Goal.")
    status: Status = Field(
        description="A variable describing the status of the Goal.")


class Plan(BaseModel):
    """\nA Plan is a list of Subgoals. Subgoals are selected so that when all Subgoals are completed, the intention of the user instruction is also fulfilled. The Subgoals should be given in an order where they can be executed sequentially in case a subgoal depends on the results of a previous subgoal."""
    Subgoals: list[Goal] = Field(description="A Plan is a list of Subgoals. Subgoals are selected so that when all Subgoals are completed, the intention of the user instruction is also fulfilled. The Subgoals should be given in an order where they can be executed sequentially in case a Subgoal depends on the results of a previous Subgoal.")
    activeSubgoalId: int = Field(
        description="The activeSubgoalID is the ID of the current or next subgoal to work on.")


class ExplanationAndInitialPlan(BaseModel):
    "Use this class to create an initial plan."
    ScreenshotObservation: str = Field(
        description="Describe what you observe in the latest image. Focus on relevant information. What menu or webpage is the device currently in? Accurate visual observations are an important foundation for the rest of your response and reasoning. Additionally, use the Screen Information Text to spot potential user interface elements.")
    StepByStepExplanation: str = Field(description=detailed_step_by_step_explanation +
                                       " creating the plan. Start by analyzing the user intention. Think about which unknown information needs to be gathered through the exploration to fulfill the user intention and add requests for those pieces of information to the ListOfDesiredInformation and the completion criteria of the corresponding subgoals.")
    ListOfDesiredInformation: list[AgentMemoryEntry] = Field(
        description="The ListOfDesiredInformation is a list of AgentMemoryEntry, pieces of information that are currently unknown and are necessary to retrieve to complete the instruction. Each AgentMemoryEntries should be mentioned in the CompletionCriteria of a Subgoal. Only add requests for information necessary for the completion of the user instruction to this list. If the user instruction requires finding n pieces of information, create an AgentMemoryEntry for each of the n pieces of information. Only combine multiple pieces of information in an AgentMemoryEntry if necessary, e.g., if the exact number of pieces of information is unknown.")
    InitialPlan: Plan = Field(description="The initial Plan that should be followed to fulfill the user instruction. For each piece of desired information that needs to be retrieved, explicitly mention adding it to the agent memory in your Plan. Only add or request information in the Agent Memory that is essential to complete the Subgoals or the user instruction.")


class ExplanationAndPlanUpdate(BaseModel):
    """Use this class to explain what happened after the Agent executed the last Action and modify the active Subgoal accordingly."""
    ScreenshotDifferenceObservation: str = Field(
        description="Accurate visual observations are an important foundation for the rest of your response and reasoning. Describe what you observe in the images. Additionally, use the Screen Information Texts to spot potential user interface elements. What menu or webpage is the device currently in? Did the menu or webpage change because of the last action? Focus on all relevant changes. What changes are the result of the last Action? Did user interface elements like buttons, text field or sliders pop up?")
    StepByStepExplanation: str = Field(description=detailed_step_by_step_explanation +
                                       "the success or failure of the last action and updating the plan, subgoals, and Agent Memory. If an Action fails, try to explain the reason for the failure. If an Action fails multiple times in a row, try to find alternative Actions to achieve the active Subgoal. Pay particular attention to the entries in the Agent Memory and the question of what AgentMemoryEntry needs to be added to the Agent Memory before the active Subgoal can be completed. Very important: If a search bar or other long element unexpectedly appeared when a slider was needed, suggest swiping on the element next to check if it is a slider, as you are terrible at recognizing sliders. If this is the case, explicitly mention the bounding box coordinates of the potential slider.")
    ActionOutcomeDescription: str = Field(
        description="A detailed description of the outcome of the last Action. Describe in detail what the Action did. Specially mention if the action made a menu or pop-up show up. Mention if the action changed the focused text field. Did the Action bring the agent closer to the completion of the active subgoal/user instruction or was the Action a failure? Explain what needs to be done to correct the failure and come back on track to fulfill the active Subgoal. Consider if an interaction with a different user interface element would have been necessary. If inputting text did nothing, reason if the correct text field was selected. Very important: If a search bar or other long element unexpectedly appeared when a slider was needed, suggest swiping on the element next to check if it is a slider, as you are terrible at recognizing sliders. If this is the case, explicitly mention the bounding box coordinates of the potential slider.")
    activeSubgoalUpdate: Goal = Field(description="Update the status of the activeSubgoal after the action. The ID must stay the same. Completion Criteria and Title should only change for good reasons, e.g., when the original plan seems impossible. Update the Progress Description and PlannedNextSteps according to the success or failure of the last action. Planned next steps refer to the next steps for the completion of the Subgoal. This should be an empty list if the Subgoal is completed. It might take multiple swipes to change the state of the user interface to achieve the desired outcome. Use the changes in the screenshot to determine after a swipe if more swipes might lead to the completion of the Subgoal or if some other approach should be tried next. If the active Subgoal becomes irrelevant to achieving the user instruction, mark it as completed. If important information is missing, before proceeding to the next Subgoal, modify the active Subgoal and the ListOfUpdatedAndAdditionallyDesiredInformation with the intention of collecting such information in the Agent Memory. Only text information can be collected.")
    ListOfUpdatedAndAdditionallyDesiredInformation: list[AgentMemoryEntry] = Field(
        description="Based on the screenshot and the last action, you can update existing entries in the Agent Memory by adding updated Entries with the same VariableName to this list. You can also add entirely new Entries, either complete information or requests for information. A new AgentMemoryEntry needs an unused VariableName.")
    nextSubgoalId: int = Field(description="Give the SubgoalId of the next Subgoal that should be worked on. This is the ID of the active Subgoal if it has not been completed yet. If the active Subgoal was completed, select the appropriate next Subgoal. If all Subgoals are completed, set this to an unused ID.")


class Memory(BaseModel):
    memory: list[AgentMemoryEntry] = []

class ActionSelection(BaseModel):
    "Observe, think, and select the next Action."
    ScreenshotObservation: str = Field(
        description="Describe what you observe in the latest image. Accurate visual observations are an important foundation for the rest of your response and reasoning. Additionally, use the Screen Information Text to spot potential user interface elements.") 
    Thought: str = Field(description=detailed_step_by_step_explanation +
                         "the chosen action. If you want to tap, long-press, or swipe, give very detailed reasoning for the selected start and end points using the screenshot and the detected bounding boxes as input. If you want to interact with text that is detected multiple times, carefully reason about your choice. \nPay special attention to text operations. Use the information about the focused text field; you have to reason if you have the correct text field selected or if you need to choose the proper text field first.")
    action: Action = Field(
        description="The Action with the correct parameters to proceed with the active Subgoal.")
    ActionDescription: str = Field(
        description="A detailed description of the chosen action and the predicted outcome. Make clear that this is only a prediction, as the actual behavior might be different. Include a short description of the state of the UI element you want to interact with before the action.")


class TestConfiguration(BaseModel):
    """All relevant settings for a single task execution are saved in this class."""

    task: str = Field(description="The task that the agent should complete.")
    model: str = Field(description="The MLLM that is used.")
    temperature: float = Field(
        default=0.0, description="The temperature that MLLM uses.")
    number: int = Field(
        default=1, description="The number of the run. Use this if you want multiple runs with otherwise identical TestConfiguration.")
    max_tokens: int = Field(
        default=8192, description="Maximum number of returned tokens.")
    max_thinking_tokens: int = Field(
        default=4096, description="Maximum number of thinking tokens.")
    max_steps: int = Field(
        default=20, description="After how many steps the agent stops if it cannot complete the task.")
    sub_folder: str = Field(description="Name of the subfolder to save the loggin information to.")
    action_vis: bool = Field(default = False, description="Visualize actions for reflections as points and arrows.")
    top_p : float = Field(default = 0.2) # I cannot believe that I did not set this before. It feels like the LLM is now listening more to the instructions.
    def generate_hash(self):
        """generate_hash is used for generating directory names. Collisions are possible but so unlikely that they are ignored.

            :return: For a TestConfiguration, generate_hash returns the first 16 digits of its hash value.
            :rtype: str
        """
        return sha256(str(self.model_dump_json()).encode('utf-8')).hexdigest()[:16]

# endregion

class InfoPool:
    """Keeping track of all information across the agents."""
    # User input / accumulated knowledge
    instruction: str = ""
    tips: str = ""
    keyboard_history: list = []
    focused_element_history: list = []
    focused_element_history_character_count: list = []
    # Perception
    width: int
    height: int
    perception_info_history: list = []  # list of textual ui descriptions
    # Working memory
    summary_history: list = []  # List of action descriptions
    action_history: list = []  # List of actions
    outcome_description_history: list = []
    # Planning
    plan: Plan = None
    memory: Memory = None
    config: TestConfiguration = None


def generate_history(info_pool):
    prompt = "### Recent Action History ###\n"
    k = 5
    if len(info_pool.outcome_description_history) == 0:
        return "No history."
    # This works as the last action is saved at the end of info_pool.summary_history
    prompt += "There is history for the "+str(len(info_pool.outcome_description_history)-max(
        0, len(info_pool.outcome_description_history)-k))+" steps: "
    for i in range(max(0, len(info_pool.outcome_description_history)-k), len(info_pool.outcome_description_history)):
        prompt += " Step " + str(i) + ": "+"Action: "+info_pool.action_history[i].model_dump_json(
        )+", Action Description and Predicted Outcome: "+info_pool.summary_history[i]+" Actual Outcome: "+info_pool.outcome_description_history[i]
        prompt += "\n"
    return prompt+"\n"


class BaseAgent(ABC):
    @abstractmethod
    def init_chat(self) -> list:
        pass

    @abstractmethod
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass


class InitialPlaner(BaseAgent):
    """This class creates part of the prompt for initial Planning."""

    def init_chat(self):
        system_prompt = helpful_android_assistant + \
            " Your objective is to create an initial plan to achieve the user's requests. "
        return ("model", [system_prompt])

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = ""
        if info_pool.tips != "":
            prompt += "### Tips ###\n"
            prompt += "The following tips that might be useful:\n"
            prompt += f"{info_pool.tips}\n\n"
        prompt += "---\n"
        prompt += "### Screen Information Text ###\n"
        prompt += info_pool.perception_info_history[-1]
        prompt += "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"
        prompt += "---\n"
        prompt += "Your task is to break down the user instruction into a sequence of concrete subgoals. If the instruction involves exploration, include concrete subgoals to quantify the investigation steps. You are given a screenshot displaying the phone's starting state.\n\n"
        prompt += action_space_string
        prompt += " Additionally, you plan to add important text information to the agent's memory. If required, explicitly mention adding and completing information to the agent memory in your plan."
        prompt += "---\n"
        prompt += "Provide your output in the specified format."
        return prompt


class Manager(BaseAgent):
    "This class creates part of the prompt for Planning, Reflection, and Notetaking."""

    def init_chat(self):
        system_prompt = helpful_android_assistant+" Your objective is to analyze the screenshots and the last action to track progress, as well as modify the active subgoal to achieve the user's requests. Actions selection is done in another submodule of the agent. You are responsible for planning, reflecting, and updating Agent Memory."
        return ("model", [system_prompt])

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "\nThe top left corner is the origin. The x and y axis go horizontally and vertically, respectively. Coordinate values are normalized to 0-1000 for every image. There are two images attached. The first image is a phone screenshot before the last action."
        if(info_pool.config.action_vis):
            if(info_pool.action_history[-1].name == ActionType.tap or info_pool.action_history[-1].name == ActionType.long_press):
                prompt += "The last action was a tap or long_press. The point of interaction is visualized by a red ring in the first image."
            elif (info_pool.action_history[-1].name == ActionType.swipe):
                prompt += "The last action was a swipe. The swipe is visualized as a red arrow in the first image."
        prompt += "\n\n"
        prompt += "### Screen Information Text Before the Action ###\n"
        prompt += info_pool.perception_info_history[-2]
        prompt += "\n"
        prompt += "### Screen Information Text After the Action ###\n"
        prompt += info_pool.perception_info_history[-1]
        prompt += "\n"
        prompt += "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"
        prompt += generate_history(info_pool)
        prompt += "The history does not yet contain information for the latest Action: "
        step = len(info_pool.action_history)-1
        prompt += " Step " + str(step) + ": "+info_pool.action_history[step].model_dump_json(
        ) + ", Action Description and Predicted Outcome: "+info_pool.summary_history[step]
        prompt += " The Actual Outcome of the latest Action is something you have to judge.\n"
        prompt += "A Plan is a list of Subgoals that can be sequentially executed and aim to fulfill the user instruction."
        prompt += "\n\n### Current Plan ###\n"
        prompt += f"{info_pool.plan.model_dump_json()}\n\n"
        prompt += "\n\n### Current Agent Memory ###\n"
        prompt += f"{info_pool.memory.model_dump_json()}\n\n"

        if info_pool.tips != "":
            prompt += "### Tips ###\n"
            prompt += "The following tips that might be useful:\n"
            prompt += f"{info_pool.tips}\n\n"
        prompt += "Carefully assess the current status to determine if the last action was successful and update the active Subgoal accordingly."
        prompt += action_space_string
        prompt += " Additionally, to enable information retrieval, you can add information and information requests to the agent's memory."
        prompt += "---\n"
        prompt += orientation_information
        prompt += " Provide your output in the specified format."
        return prompt


class ActionSelector(BaseAgent):
    "This class creates part of the prompt for Action Selection."""

    def init_chat(self):
        system_prompt = helpful_android_assistant + \
            " Your objective is to choose the correct next action to come closer to the completion of the user's instruction. "
        return ("model", [system_prompt])

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"
        prompt += "A Plan is a list of Subgoals that can be sequentially executed and aim to fulfill the user instruction."
        prompt += "\n\n### Current Plan ###\n"
        prompt += f"{info_pool.plan.model_dump_json()}\n\n"
        prompt += "\n\n### Current Agent Memory ###\n"
        prompt += f"{info_pool.memory.model_dump_json()}\n\n"
        prompt += generate_history(info_pool)
        prompt += "### Screen Information Text ###\n"
        prompt += info_pool.perception_info_history[-1]
        prompt += "\n"
        prompt += "Note that a search bar is often a long, rounded rectangle. If no search bar is presented and you want to perform a search, you may need to tap a search button, which is commonly represented by a magnifying glass.\n"
        prompt += "You are given a screenshot of the device screen. This is your most important source of information."
        prompt += "\n\n"
        if info_pool.tips != "":
            prompt += "### Tips ###\n"
            prompt += "The following tips might be useful:\n"
            prompt += f"{info_pool.tips}\n\n"
        prompt += "---\n"
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice an unsolved error in the previous action, attempt to rectify it."
        prompt += "### Tips for Swiping ###\n"
        prompt += "Swipe from position [start_y, start_x] to position [end_y, end_x]. To swipe up or down to review more content, you can adjust the y-coordinate offset based on the desired scroll distance. If possible, start swipes at the center of objects you want to interact with in order to simulate human behavior more realistically."
        if (info_pool.keyboard_history[-1]):
            prompt += "The virtual keyboard is open. You can input or delete text."
        else:
            prompt += "The virtual keyboard is currently not visible. To input or delete text, first select the correct text field."
        prompt += "\nHINT: If multiple tap actions fail to make changes to the screen, consider using a \"swipe\" action to view more content or use another way to achieve the current subgoal."
        prompt += "---\n"
        prompt += orientation_information
        prompt += " Provide your output in the specified format."
        return prompt
