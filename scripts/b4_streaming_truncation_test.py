#!/usr/bin/env python3
"""Appendix B.4: Streaming truncation test (run against a live endpoint)

From The LLM Inference Handbook, Volume II. See README.md, Appendix B, for
the setup (tokenizer files under tok/<name>/, corpora under data/).
"""
# Sends diacritized Arabic prompts with odd max_tokens values and checks every
# streamed delta for U+FFFD and for prompt leakage in the first delta.
import requests, json

URL = "http://localhost:8000/v1/chat/completions"
PROMPT = "اكتب فقرة مشكّلة بالكامل عن أهمية القراءة."
problems = 0
for max_tokens in [7, 13, 21, 33, 47, 61, 83, 101]:
    r = requests.post(URL, json={"model": "served-model", "stream": True, "max_tokens": max_tokens,
                                 "messages": [{"role": "user", "content": PROMPT}]}, stream=True)
    deltas = []
    for line in r.iter_lines():
        if line.startswith(b"data: ") and line != b"data: [DONE]":
            chunk = json.loads(line[6:])
            d = chunk["choices"][0]["delta"].get("content")
            if d:
                deltas.append(d)
    text = "".join(deltas)
    if "�" in text:
        problems += 1; print(f"max_tokens={max_tokens}: U+FFFD in output")
    if deltas and PROMPT[:10] in deltas[0]:
        problems += 1; print(f"max_tokens={max_tokens}: prompt leaked into first delta")
print("problems:", problems)
