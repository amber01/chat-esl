from typing import List, Dict, Generator
from openai import OpenAI


class LLMClient:
    """封装OpenAI兼容的模型客户端"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def stream_chat(self, messages: List[Dict], temperature: float = 0.7) -> Generator[str, None, None]:
        """流式聊天"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
