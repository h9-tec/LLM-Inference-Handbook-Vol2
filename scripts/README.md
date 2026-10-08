# Measurement scripts

These are the Appendix B scripts from the handbook, extracted so you can run them directly.

| Script | Reproduces | Needs |
|---|---|---|
| `b1_fertility_normalization_escaping.py` | Chapter 20 tables: tokens per word, classic normalization, Eastern digits, JSON escaping | tokenizer files, `data/wiki_ar.txt`, `data/tashkeela_sample.txt` |
| `b2_nfc_shadda_prefix_divergence.py` | Chapter 20.2: NFC reordering and where the token sequence diverges | tokenizer files, `data/abdou_vocalized_sample.txt` |
| `b3_partial_utf8_tokens.py` | Chapter 21.1: share of tokens that are partial UTF-8 | tokenizer files, both corpora |
| `b4_streaming_truncation_test.py` | Chapter 21.1 test: odd `max_tokens` against a live endpoint | a running OpenAI-compatible server |

Setup:

```bash
pip install tokenizers tiktoken sentencepiece pyarrow huggingface_hub requests
mkdir -p tok data
# tokenizer.json for each model under tok/<name>/, e.g. tok/qwen3/tokenizer.json
# ALLaM uses SentencePiece: tok/allam/tokenizer.model
```

Corpora used for the numbers in the book: a 33k-word Arabic Wikipedia sample, and 3,000-line samples from `arbml/tashkeela` and `Abdou/arabic-tashkeel-dataset` on Hugging Face. Rerun on your own corpus before trusting any number for your deployment.
