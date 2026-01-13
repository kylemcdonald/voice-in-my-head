import openai
import json
import random
import csv
from time import time
from helpers import log

def build_dialog(messages, mapping={"user": "Patient", "assistant": "Therapist"}):
    dialog = []
    for m in messages:
        role = mapping[m["role"]]
        content = m["content"]
        dialog.append(f"{role}: {content}")
    return "\n".join(dialog)

class ChatGPT:
    def __init__(self, model="gpt-5.2", language="en"):
        self.openai = openai.OpenAI()
        self.model = model
        self.language = language
        self.load_localized_strings()

    def load_localized_strings(self):
        self.strings = {}
        with open('scripts/chatgpt.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.strings[row['Key']] = row[self.language]

    def chatgpt(self, prompt, system=None, backup=None):
        messages = []
        messages.append({"role": "user", "content": prompt})
        log({"prompt": prompt})
        if system:
            messages.insert(0, {"role": "system", "content": system})
            log({"system": system})
        try:
            log({"chatgpt": messages})
            start_time = time()
            result = self.openai.chat.completions.create(
                model=self.model, messages=messages, max_completion_tokens=1024
            )
            request_duration = time() - start_time
            content = result.choices[0].message.content
            log({"chatgpt": {"result": content, "duration": request_duration}})
            return content
        except openai.InternalServerError as e:
            log({"chatgpt-error": e})
            return backup

    def convert_response_to_name(self, response):
        if response == "":
            return self.strings['convert_response_to_name_backup']
        return self.chatgpt(
            self.strings['convert_response_to_name_prompt'].format(response=response),
            system=self.strings['convert_response_to_name_system'],
            backup=self.strings['convert_response_to_name_backup'],
        )

    def convert_existing_to_summary(self, messages):
        dialog = build_dialog(messages)
        return self.chatgpt(
            self.strings['convert_existing_to_summary_prompt'].format(dialog=dialog),
            system=self.strings['convert_existing_to_summary_system'],
            backup=self.strings['convert_existing_to_summary_backup'],
        )

    def convert_goals_to_summary(self, messages):
        dialog = build_dialog(messages)
        return self.chatgpt(
            self.strings['convert_goals_to_summary_prompt'].format(dialog=dialog),
            system=self.strings['convert_goals_to_summary_system'],
            backup=self.strings['convert_goals_to_summary_backup'],
        )

    def convert_goals_to_summary_prompt(self, messages):
        dialog = build_dialog(messages)
        return self.chatgpt(
            self.strings['convert_goals_to_summary_prompt_prompt'].format(dialog=dialog),
            system=self.strings['convert_goals_to_summary_prompt_system'],
            backup=self.strings['convert_goals_to_summary_prompt_backup'],
        )

    def respond_to_overheard(self, overheard):
        if not overheard or overheard == "":
            kind = random.choice(self.strings["respond_to_overheard_kinds"].split(","))
            verb = random.choice(self.strings["respond_to_overheard_verbs"].split(","))
            return self.chatgpt(
                self.strings['respond_to_overheard_empty_prompt'].format(kind=kind, verb=verb),
                system=self.strings['respond_to_overheard_system'].format(goals_prompt=self.goals_prompt),
                backup=self.strings['respond_to_overheard_backup'],
            )
        return self.chatgpt(
            self.strings['respond_to_overheard_prompt'].format(overheard=overheard),
            system=self.strings['respond_to_overheard_system'].format(goals_prompt=self.goals_prompt),
            backup=self.strings['respond_to_overheard_backup'],
        )

    def convert_experience_to_memory(self, entire_transcript):
        return self.chatgpt(
            self.strings['convert_experience_to_memory_prompt'].format(entire_transcript=entire_transcript),
            system=self.strings['convert_experience_to_memory_system'].format(goals_prompt=self.goals_prompt),
            backup=self.strings['convert_experience_to_memory_backup'],
        )
