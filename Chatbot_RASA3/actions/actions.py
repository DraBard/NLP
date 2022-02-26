# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_core_sdk.events import Restarted
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionDenial(Action):
#
#     def name(self) -> Text:
#         return "action_denial"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#
#         return [Restarted()]
#
# class ActionRestart(Action):
#
#   def name(self) -> Text:
#       return "action_restart"
#
#   def run(self, dispatcher, tracker, domain):
#       # do something here
#
#       return []
