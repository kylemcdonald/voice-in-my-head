import openai
import json
import random
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
    def __init__(self, model="gpt-4-1106-preview"):
        self.openai = openai.OpenAI()
        self.model = model

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
                model=self.model, messages=messages, max_tokens=1024
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
            return "friend"
        return self.chatgpt(
            f'I asked someone for their name, and they said "{response}". What should I call them? If it seems like they didn\t give a name, just say "friend".',
            system="You are a helpful assistant that always responds by giving a single name, with no additional commentary or punctuation.",
            backup="friend",
        )

    def convert_existing_to_summary(self, messages):
        dialog = build_dialog(messages)
        return self.chatgpt(
            f"""Here is a previous discussion between us about how my internal monologue works:

{dialog}

Please summarize my responses in a way that helps me reflect on my answers, beginning with "It sounds like the voice in your head is...".""",
            system="You are my helpful personal AI, I talk with you in order to understand myself better.",
            backup="It sounds like the voice in your head is beautiful but complicated.",
        )

    def convert_goals_to_summary(self, messages):
        dialog = build_dialog(messages)
        return self.chatgpt(
            f"""Here is a previous discussion between us about how I wish my internal monologue worked:

{dialog}

Please summarize my responses into a single paragraph, in a way that helps me reflect on my answers, beginning with "It sounds like you want the voice in your head to be...".""",
            system="You are my helpful personal AI, I talk with you in order to understand myself better.",
            backup="Hmm, it sounds like you want the voice in your head to be strong, and kind.",
        )

    def convert_goals_to_summary_prompt(self, messages):
        dialog = build_dialog(messages)
        return self.chatgpt(
            f"""Here is a previous discussion between us about how I wish my internal monologue worked:

{dialog}

Please summarize all my responses in three sentences, written from my perspective, as an affirmation, beginning with "The Voice In My Head is...".""",
            system="You are my helpful personal AI, I talk with you in order to understand myself better.",
            backup="The voice in my head is strong and kind.",
        )

    def respond_to_overheard(self, overheard):
        if not overheard or overheard == "":
            kind = random.choice(
                ["life", "conversational", "emotional", "inter-personal"]
            )
            verb = random.choice(
                ["think", "feel", "will", "want to", "need to", "hope that", "wish"]
            )
            return self.chatgpt(
                f'Share some broadly useful {kind} advice, in one sentence, as the Voice In My Head. Start with a phrase like "I {verb}".',
                system=f"You are a helpful Voice In My Head. {self.goals_prompt}. You always respond in the first person, as if you are me.",
                backup="Hmm, how interesting.",
            )
        return self.chatgpt(
            f"""We just overheard this conversation between myself and others: "{overheard}"

Respond to this in-character, in one sentence, as the Voice In My Head.""",
            system=f"You are a helpful Voice In My Head. {self.goals_prompt}. You always respond in the first person, as if you are me.",
            backup="Hmm, how interesting.",
        )

    def convert_experience_to_memory(self, entire_trancript):
        return self.chatgpt(
            f"""Here is a long transcript of things we overheard ourselves and others saying over the last half hour: "{entire_trancript}".

As the Voice In My Head, please identify one particular phrase that stood out to you. Explain why you found the phrase interesting, and then ask me to repeat after you, and then say the phrase.

For example, if you identify the phrase \"Anything can happen next\", then you might say something like: \"In reflecting on what just transpired, there is one phrase that stuck. I hope you will hold it as a memory of this time together. I heard something that made me reflect on the complexity of life, and the strangeness of our present moment. I would like to share this phrase with you. Are you ready? Repeat after me: 'Anything can happen next.'\"""",
            system=f"You are a helpful Voice In My Head. {self.goals_prompt}. You always respond in the first person, as if you are me.",
            backup="I feel lucky to have spent the day with you. Please remember one moment that you are grateful for, and take it with you.",
        )
