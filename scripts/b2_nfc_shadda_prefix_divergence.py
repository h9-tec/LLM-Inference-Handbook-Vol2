#!/usr/bin/env python3
"""Appendix B.2: NFC shadda reordering and prefix divergence

From The LLM Inference Handbook, Volume II. See README.md, Appendix B, for
the setup (tokenizer files under tok/<name>/, corpora under data/).
"""
import unicodedata, re
from tokenizers import Tokenizer

diac_lines = open("data/abdou_vocalized_sample.txt", encoding="utf-8").read().split("\n")
SHADDA = 'ّ'
VOWEL = '[ً-ِْ]'
txt = "\n".join(diac_lines)
print("shadda->vowel (typed order):", len(re.findall(SHADDA + VOWEL, txt)),
      " vowel->shadda (NFC order):", len(re.findall(VOWEL + SHADDA, txt)))

changed_lines = sum(unicodedata.normalize("NFC", l) != l for l in diac_lines) / len(diac_lines)
words = txt.split()
changed_words = sum(unicodedata.normalize("NFC", w) != w for w in words) / len(words)
print(f"NFC changes {changed_lines:.0%} of lines and {changed_words:.0%} of words")

for name in ["llama3", "qwen3", "gemma3", "deepseek-v3", "falcon-h1", "fanar"]:
    tk = Tokenizer.from_file(f"tok/{name}/tokenizer.json")
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids
    fracs = []
    for l in diac_lines[:800]:
        x, y = enc(l), enc(unicodedata.normalize("NFC", l))
        if x != y:
            i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            fracs.append(i / len(x))
    import json
    normalizer = json.load(open(f"tok/{name}/tokenizer.json")).get("normalizer")
    mean_div = sum(fracs) / len(fracs) if fracs else float('nan')
    print(f"{name}: normalizer={normalizer} | lines diverging={len(fracs)}/800 | "
          f"mean first-divergence at {mean_div:.0%} of line")
