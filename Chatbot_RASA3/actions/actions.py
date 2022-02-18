# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.forms import FormAction

# class ActionBookingForm(FormAction):
#
#     def name(self) -> Text:
#         return "action_guests_number"
#
#     @staticmethod
#     def required_slots(tracker: "Tracker") -> List[text]:
#         return ["n_ppl"]
#
#     def submit(
#         self,
#         dispatcher: "CollectingDispatcher",
#         tracker: "Tracker",
#         domain: "DomainDict",
#     ) -> List[EventType]:
#
#         n_ppl = tracker.get_slot('n_ppl')
#
#         print(n_ppl)
#
#         dispatcher.utter_message(text = f"Splendid! You have a room booked for {n_ppl} people with breakfast. Payment method: credit card at reception. Can you confirm please?")
#         return []