#!/usr/bin/env python3
"""Appendix B.3: Partial-UTF-8 tokens (streaming split risk)

From The LLM Inference Handbook, Volume II. See README.md, Appendix B, for
the setup (tokenizer files under tok/<name>/, corpora under data/).
"""
import collections, unicodedata
from tokenizers import Tokenizer
import tiktoken

def partial_share_hf(name, text):
    tk = Tokenizer.from_file(f"tok/{name}/tokenizer.json")
    enc = tk.encode(text, add_special_tokens=False)
    bad = sum('�' in tk.decode([i]) for i in enc.ids)
    cov = collections.Counter()
    for (s, e) in enc.offsets:
        for k in range(s, e):
            cov[k] += 1
    split_chars = collections.Counter(text[k] for k, v in cov.items() if v > 1)
    total = sum(split_chars.values())
    print(f"{name}: partial tokens {bad/len(enc.ids):.2%} | "
          f"1 split char per {len(text.split())/max(total,1):.0f} words | "
          f"top: {[(unicodedata.name(c, '?'), n) for c, n in split_chars.most_common(4)]}")

def partial_share_tiktoken(encname, text):
    e = tiktoken.get_encoding(encname)
    ids = e.encode(text, disallowed_special=())
    bad = 0
    for i in ids:
        try:
            e.decode_single_token_bytes(i).decode()
        except UnicodeDecodeError:
            bad += 1
    print(f"{encname}: partial tokens {bad/len(ids):.2%}")

for corpus_name, path in [("MSA", "data/wiki_ar.txt"), ("diacritized", "data/tashkeela_sample.txt")]:
    text = open(path, encoding="utf-8").read()[:200000]
    print(f"--- {corpus_name} ---")
    for n in ["qwen3", "llama3", "gemma3", "deepseek-v3", "falcon-h1", "fanar"]:
        partial_share_hf(n, text)
    for en in ["o200k_base", "cl100k_base"]:
        partial_share_tiktoken(en, text)
