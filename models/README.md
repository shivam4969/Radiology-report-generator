# models/

Exported ExecuTorch programs go here and are gitignored:

- `cxr-densenet121.pte` + `.meta.json` from `python scripts/export_cxr.py` (XNNPACK) or `--backend qnn`
- `llama3_2-3b-*.pte` / `qwen3-4b-*.pte` and tokenizer from `scripts/export_llm.md`

Never commit weights, `.pte` files or anything from the Qualcomm SDK.
