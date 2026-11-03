"""
Language model access for VIMH.

Two model slots are configured in .env:
- LLM_MODEL / LLM_REASONING: interjections, summaries and the closing memory
- LLM_FAST_MODEL / LLM_FAST_REASONING: interview calls the visitor waits on

The provider is picked from the model name ("claude-..." uses Anthropic with
CLAUDE_API_KEY, "gpt-..." uses OpenAI with OPENAI_API_KEY). Reasoning is
"off", "low", "medium" or "high".
"""

import os
from typing import Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

REASONING_LEVELS = ("off", "low", "medium", "high")

# Claude models that turn thinking off with a different value than "disabled".
CLAUDE_THINKING_OFF = {
    "claude-sonnet-5-5": {"type": "between_tools"},
}


class LLM:
    def __init__(self, model: str, reasoning: str):
        if reasoning not in REASONING_LEVELS:
            raise ValueError(f"Reasoning must be one of {REASONING_LEVELS}, got {reasoning!r}")
        self.model = model
        self.reasoning = reasoning
        if model.startswith("claude-"):
            self.provider = "anthropic"
            self._client = AsyncAnthropic(api_key=os.environ["CLAUDE_API_KEY"])
        elif model.startswith("gpt-"):
            self.provider = "openai"
            self._client = AsyncOpenAI()
        else:
            raise ValueError(f"Don't know which provider serves {model!r}")

    async def complete(self, prompt: str, system: Optional[str] = None) -> str:
        if self.provider == "anthropic":
            return await self._complete_anthropic(prompt, system)
        return await self._complete_openai(prompt, system)

    async def _complete_anthropic(self, prompt: str, system: Optional[str]) -> str:
        if self.reasoning == "off":
            extra = {"thinking": CLAUDE_THINKING_OFF.get(self.model, {"type": "disabled"})}
        else:
            extra = {"thinking": {"type": "adaptive"}, "output_config": {"effort": self.reasoning}}
        kwargs = {"system": system} if system else {}
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra,
            **kwargs,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def _complete_openai(self, prompt: str, system: Optional[str]) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=self.model,
            reasoning_effort="none" if self.reasoning == "off" else self.reasoning,
            messages=messages,
        )
        return response.choices[0].message.content or ""


def from_env(fast: bool = False) -> LLM:
    prefix = "LLM_FAST_" if fast else "LLM_"
    return LLM(os.environ[prefix + "MODEL"], os.environ[prefix + "REASONING"])
