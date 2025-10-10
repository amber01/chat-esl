from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatSession:
    messages: List[Message] = field(default_factory=list)

    def add_user(self, text: str):
        self.messages.append(Message(role="user", content=text))

    def add_assistant(self, text: str):
        self.messages.append(Message(role="assistant", content=text))

    def to_openai(self, system_prompt: str) -> List[Dict]:
        data = [{"role": "system", "content": system_prompt}]
        for m in self.messages:
            data.append({"role": m.role, "content": m.content})
        return data


class ConversationManager:
    """管理多个会话实例"""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession()
        return self.sessions[session_id]
