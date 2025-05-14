from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action

from pathlib import Path

from io_manager import IOManager
from utility import *
from prompts import TestConfiguration
from run_task import do_task



class BachelorAgent2(base_agent.EnvironmentInteractingAgent):
  """Human agent; wait for user to indicate they are done."""
  def __init__(
      self,
      env: interface.AsyncEnv,
      name: str = '',
      transition_pause: float | None = 1.0,
  ):
    super().__init__(env, name, transition_pause)
    p = Path(__file__).resolve().parent
    folder = "task"
    self.inout = IOManager(base_directory=p/folder, verbose=False,
                      overwrite=True, wait_between_calls=False)
    model = "gemini-2.5-pro-preview-05-06"
    sub_folder = "4"
    self.test_config = TestConfiguration(task="", model=model, sub_folder=sub_folder, action_vis= True)

  def step(self, goal: str, n = 0) -> base_agent.AgentInteractionResult:
    self.inout.answer = ""
    self.test_config.task = goal
    b = self.inout.base_directory
    print(b)
    print("This should be the expected number of steps:", n)
    m = 2*n
    print("Running agent for:" + str(m) +"steps.")
    self.test_config.max_steps = m
    self.inout.base_directory= b/self.test_config.sub_folder
    try:
      do_task(self.inout, self.test_config)
    except Exception as e:
      print("Something went wrong: --> "+ str(e)+" <--")
    print("Current costs are: ")
    self.inout.estimate_cost(self.test_config.model)
    print("---")
    self.inout.base_directory = b
    response = self.inout.answer
    action_details = {'action_type': 'answer', 'text': response}
    self.env.execute_action(json_action.JSONAction(**action_details))

    state = self.get_post_transition_state()
    result = {}
    result['elements'] = state.ui_elements
    result['pixels'] = state.pixels
    return base_agent.AgentInteractionResult(True, result)

  def get_post_transition_state(self) -> interface.State:
    return self.env.get_state()
