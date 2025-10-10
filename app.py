import os
import json
import time
import threading
import queue
import streamlit as st
import websockets
from utils import safe_getenv, now_ts
from core.modes import MODE_PROMPTS

APP_TITLE = "Chat-ESL · English Tutor"
DEFAULT_MODEL = safe_getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = safe_getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
BACKEND_WS = os.getenv("BACKEND_WS", "ws://localhost:8000/ws")

# 状态初始化
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{now_ts()}"
if "buffer" not in st.session_state:
    st.session_state.buffer = ""

st.set_page_config(page_title=APP_TITLE, page_icon="🗣️")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("🔑 API配置")
    provider = st.selectbox("Provider", ["OpenAI", "OpenRouter", "Custom"])
    base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
    api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.divider()
    st.subheader("🎛️ 教学模式")
    mode = st.radio("Mode", list(MODE_PROMPTS.keys()), index=0)
    native_lang_aid = st.checkbox("Explain in Chinese", value=True)
    gentle_corrections = st.checkbox("Gentle corrections", value=True)
    st.divider()
    st.text_input("Backend WS", value=BACKEND_WS, key="_backend_ws")

# 展示历史
for m in st.session_state.history:
    with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "🎓"):
        st.markdown(m["content"])

user_input = st.chat_input("Type your English here… (你也可以中文)")

recv_q: "queue.Queue[str]" = queue.Queue()


def ws_worker(cfg, text):
    async def _run():
        uri = st.session_state._backend_ws
        async with websockets.connect(uri, max_size=2**23) as ws:
            await ws.send(json.dumps(cfg))
            await ws.send(json.dumps({"text": text}))
            async for msg in ws:
                recv_q.put(msg)
    import asyncio
    asyncio.run(_run())


if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🎓"):
        placeholder = st.empty()
        st.session_state.buffer = ""

        cfg = {
            "session_id": st.session_state.session_id,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "mode": mode,
            "gentle_corrections": gentle_corrections,
            "native_lang_aid": native_lang_aid,
        }

        t = threading.Thread(target=ws_worker, args=(cfg, user_input), daemon=True)
        t.start()

        start = time.time()
        while True:
            try:
                msg = recv_q.get(timeout=0.1)
                data = json.loads(msg)
                if data.get("type") == "delta":
                    st.session_state.buffer += data.get("token", "")
                    placeholder.markdown(st.session_state.buffer)
                elif data.get("type") == "done":
                    st.session_state.history.append({"role": "assistant", "content": st.session_state.buffer})
                    break
            except queue.Empty:
                if time.time() - start > 60:
                    placeholder.warning("Response timeout")
                    break

st.caption("Backend: FastAPI + WS | Frontend: Streamlit | Supports any OpenAI-compatible API.")
