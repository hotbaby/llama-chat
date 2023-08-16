# encoding: utf8

import os
import json
import streamlit as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from typing import List, Dict, Tuple, Optional

from generation import chat

model_name =  os.environ.get("MODEL_NAME", "Llama")
model_path = os.environ.get("MODEL_PATH", "/data/models/llama/llama-2-7b-chat-hf")

user_avatar = "🧑‍💻"
assistant_avatar = "🤖"


st.set_page_config(page_title=model_name)
st.title(model_name)
max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 2048, 1024, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.6, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.9, step=0.01)


@st.cache_resource
def init_model():
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar=assistant_avatar):
        st.markdown(f"Welcome to {model_name}")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = user_avatar if message["role"] == "user" else assistant_avatar
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()
    messages = init_chat_history()
    config = {
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }
    generation_config = GenerationConfig(**config)

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)

        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)

        with st.chat_message("assistant", avatar=assistant_avatar):
            placeholder = st.empty()
            for response in chat(model, tokenizer, messages, stream=True, generation_config=generation_config):
                placeholder.markdown(response)

        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
