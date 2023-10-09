# encoding: utf8

from queue import Queue
from threading import Thread
from typing import List, Dict

import torch
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
from transformers import GenerationConfig


def build_chat_input(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[Dict],
    max_new_tokens: int = 0,
    model_max_length: int = 4096,
):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue

            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)

        if round:
            rounds.append(round)

        return system, rounds

    system, rounds = _parse_messages(messages, split_role="user")
    # system tokens
    system_tokens = tokenizer.encode(system, add_special_tokens=True)

    # query tokens
    assert rounds[-1][0]["role"] == "user"
    query = "[Round {}]\n\n问：{}\n\n答：".format(len(rounds), rounds[-1][0]["content"])
    query_tokens = tokenizer.encode(query, add_special_tokens=False)

    # history tokens
    residue_tokens_length = model_max_length - max_new_tokens - len(query_tokens) - len(system_tokens)
    history = []
    for i, round in enumerate(rounds[:-1]):
        assert round[0]["role"] == "user"
        assert round[1]["role"] == "assistant"
        history.append("[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i+1, round[0]["content"], round[1]["content"]))

    history_tokens = []
    for round in history:
        round_tokens = tokenizer.encode(round, add_special_tokens=False)
        if residue_tokens_length < len(round_tokens):
            break
        residue_tokens_length -= len(round_tokens)
        history_tokens += round_tokens

    input_tokens = system_tokens + history_tokens + query_tokens
    return torch.LongTensor([input_tokens]).to(model.device)


class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(
                self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


def chat(model: LlamaForCausalLM,
         tokenizer: LlamaTokenizer,
         messages: List[dict],
         stream=True,
         generation_config: GenerationConfig = None):
    input_ids = build_chat_input(model, tokenizer, messages, generation_config.max_new_tokens)
    print(f"input: {tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=True)}")
    if stream:
        streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        Thread(target=model.generate, kwargs=dict(
            inputs=input_ids, streamer=streamer,
            generation_config=generation_config,
        )).start()
        return streamer
    else:
        outputs = model.generate(input_ids, generation_config=generation_config)
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response
