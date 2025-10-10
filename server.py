import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from utils import safe_getenv
from core.llm_client import LLMClient
from core.chat_manager import ConversationManager
from core.modes import SYSTEM_PROMPT_BASE, MODE_PROMPTS

app = FastAPI(title="Chat-ESL Backend")
conversations = ConversationManager()


class ChatConfig(BaseModel):
    session_id: str
    api_key: str | None = None
    base_url: str | None = None
    model: str
    mode: str = "Free Chat"
    gentle_corrections: bool = True
    native_lang_aid: bool = True


@app.get("/health")
async def health():
    return {"ok": True}


@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        cfg_json = await ws.receive_text()
        cfg = ChatConfig.model_validate_json(cfg_json)

        api_key = cfg.api_key or safe_getenv("OPENAI_API_KEY")
        base_url = cfg.base_url or safe_getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            await ws.send_text(json.dumps({"type": "error", "msg": "Missing API key"}))
            await ws.close()
            return

        system_prompt = SYSTEM_PROMPT_BASE
        if cfg.gentle_corrections:
            system_prompt += " Be extra supportive and positive."
        if cfg.native_lang_aid:
            system_prompt += " When user seems confused, add a short Chinese gloss."
        system_prompt += " " + MODE_PROMPTS.get(cfg.mode, "")

        client = LLMClient(api_key=api_key, base_url=base_url, model=cfg.model)
        session = conversations.get(cfg.session_id)

        while True:
            try:
                msg = await ws.receive_text()
            except WebSocketDisconnect:
                break

            data = json.loads(msg)
            user_text = data.get("text", "").strip()
            if not user_text:
                continue

            session.add_user(user_text)
            await ws.send_text(json.dumps({"type": "ack"}))

            full = ""
            for token in client.stream_chat(session.to_openai(system_prompt)):
                if token:
                    full += token
                    await ws.send_text(json.dumps({"type": "delta", "token": token}))

            session.add_assistant(full)
            await ws.send_text(json.dumps({"type": "done"}))
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "msg": str(e)}))
        await ws.close()
