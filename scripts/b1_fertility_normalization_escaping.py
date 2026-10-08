#!/usr/bin/env python3
"""Appendix B.1: Fertility, normalization, digits, and JSON escaping

From The LLM Inference Handbook, Volume II. See README.md, Appendix B, for
the setup (tokenizer files under tok/<name>/, corpora under data/).
"""
import re, json, unicodedata, collections
import tiktoken, sentencepiece as spm
from tokenizers import Tokenizer

def encoder(name):
    if name == "o200k":
        e = tiktoken.get_encoding("o200k_base")
        return lambda s: e.encode(s, disallowed_special=())
    if name == "cl100k":
        e = tiktoken.get_encoding("cl100k_base")
        return lambda s: e.encode(s, disallowed_special=())
    if name == "allam":
        sp = spm.SentencePieceProcessor(model_file="tok/allam/tokenizer.model")
        return sp.encode
    tk = Tokenizer.from_file(f"tok/{name}/tokenizer.json")
    return lambda s: tk.encode(s, add_special_tokens=False).ids

TOKENIZERS = ["o200k", "cl100k", "llama3", "qwen3", "gemma3",
              "deepseek-v3", "falcon-h1", "allam", "fanar", "bge-m3"]

# MSA corpus: keep lines that are mostly Arabic script
msa = open("data/wiki_ar.txt", encoding="utf-8").read()
msa = "\n".join(l for l in msa.split("\n")
                if len(l) > 40 and sum('؀' <= c <= 'ۿ' for c in l) / len(l) > 0.6)[:200000]

# Diacritized corpus: one fully vocalized line per row
diac = open("data/tashkeela_sample.txt", encoding="utf-8").read()
DIAC_RE = re.compile('[ً-ْ]')
undiac = DIAC_RE.sub('', diac)

classic_norm = lambda s: re.sub('[أإآ]', 'ا', s).replace('ى', 'ي').replace('ة', 'ه')
east = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
nums = " ".join(str(n) for n in [2024, 1445, 3.5, 120000, 17, 99, 2030, 750, 48, 1000000, 365, 12, 5, 88, 1979] * 20)
json_sample = msa[:50000]

def tpw(enc, text):
    return len(enc(text)) / len(text.split())

print("tokenizer | MSA t/w | undiac t/w | full-diac t/w | classic-norm delta% | east-digit x | json-escape x")
for name in TOKENIZERS:
    enc = encoder(name)
    norm_delta = (len(enc(classic_norm(msa))) - len(enc(msa))) / len(enc(msa)) * 100
    digit_x = len(enc(nums.translate(east))) / len(enc(nums))
    esc_x = len(enc(json.dumps(json_sample))) / len(enc(json.dumps(json_sample, ensure_ascii=False)))
    print(f"{name} | {tpw(enc, msa):.2f} | {tpw(enc, undiac):.2f} | {tpw(enc, diac):.2f} | "
          f"{norm_delta:+.1f} | {digit_x:.2f} | {esc_x:.1f}")
