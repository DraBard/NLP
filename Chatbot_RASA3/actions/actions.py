# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_core_sdk.events import Restarted
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.events import AllSlotsReset
#
#
# class ActionEnd(Action):
#     def name(self):
#         return "action_reset_all_slots"
#
#     def run(self, dispatcher, tracker, domain):
#         return [AllSlotsReset()]