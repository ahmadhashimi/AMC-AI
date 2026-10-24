# LitGPT connection: boilerplate without someone else's pretrained weights

Official upstream: https://github.com/Lightning-AI/litgpt
Official pretraining tutorial: https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md

## GitHub plan
1. Fork `Lightning-AI/litgpt` on GitHub into your chosen account or organization (fork is a separate training-engine codebase).
2. Keep `amc-dari-foundation` as a separate repository for data policy, configurations, reproducible experiments, and educational code.
3. Record the LitGPT commit SHA; don't blindly update during long GPU training runs.

## Verified upstream smoke test with built-in example

```bash
pip install 'litgpt[all]'
litgpt pretrain pythia-14m \
  --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/pretrain/debug.yaml
```

The official example creates randomly initialized model weights for pretraining. It uses upstream demo data. The training framework and architecture configuration are borrowed, **not pretrained model weights**.

## Official custom text path (for small corpora)

```bash
# Official docs' tokenizer-only download: does NOT download pretrained model weights
litgpt download EleutherAI/pythia-14m --tokenizer_only true

# IMPORTANT: --initial_checkpoint_dir is deliberately absent
litgpt pretrain pythia-14m \
  --tokenizer_dir EleutherAI/pythia-14m \
  --data TextFiles \
  --data.train_data_path path/to/reviewed_train_texts \
  --train.lr_warmup_steps=200 \
  --optimizer AdamW \
  --optimizer.lr 0.005
```

This is a documented demonstration, **not the final AMC tokenizer**. For AMC's trained BPE tokenizer and custom vocabulary, use the separate PyTorch learning pipeline included in this repository until compatible LitGPT tokenizer/config wiring has been implemented and tested.

For large corpora use LitData or streaming preparation instead of loading thousands of tiny `.txt` files.

## Explicitly forbidden during AMC 'from-scratch' runs
Do NOT pass `--initial_checkpoint_dir` to LitGPT unless intentionally resuming AMC's *own* previously trained weights, which should be documented as a continuation. Never use a downloaded base-model checkpoint in the from-scratch experiment.
