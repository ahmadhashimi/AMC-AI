# Wikipedia training and the epoch 2 decision

## Current experiment

- Data: `data/tokenized/wikipedia_fa_clean_v4/`.
- Training: 219,761,824 token IDs; validation: 2,265,432 token IDs.
- Tokenizer: `artifacts/wikipedia_fa_v1/tokenizer.json`, 8,192 entries.
- Model: `configs/tiny.json`, with vocabulary set from the tokenizer; 1,660,032 parameters. Starts from random weights, not pretrained weights.
- Defaults: batch size 4, sequence length 128, AdamW learning rate 0.0003, seed 42.
- One epoch covers all 219,761,823 next-token targets exactly once in shuffled windows: 429,223 optimizer steps. The final short window is padded and its padding targets are excluded from loss. Windows can cross article boundaries, marked by EOS.
- The first token is context rather than a next-token target. Test text is reserved for final evaluation.

The user reported a separate 100-step pilot: training loss 9.0291 to 7.2062; sampled validation loss 8.9511 to 7.0886. It demonstrated an operational training path, not model usefulness. The epoch run starts fresh and does not resume that pilot.

## Start, stop and resume

Run from the repository root:

```bash
.venv/bin/python -u scripts/train_epochs.py --epochs 1
```

The progress bar shows completed steps, rate, estimated remaining time and periodic training loss. `--epochs` is the total target number of epochs, not additional epochs.

Checkpoint: `outputs/wikipedia_v4_epochs/last.pt`. It includes model weights, optimizer state, epoch and next-batch position, random generator state, training settings, and dataset/tokenizer fingerprints. It saves every 1,000 steps and at epoch completion, using a temporary file followed by replacement.

Press Ctrl+C once and wait for the safe-save message. Resume with:

```bash
.venv/bin/python -u scripts/train_epochs.py --epochs 1 --resume
```

Resume checks dataset fingerprints, tokenizer, configuration, core training settings and device type. A crash or forced kill can lose work since the last checkpoint. Checkpoints are local and excluded from Git. Periodic saves replace `last.pt`; they do not create a permanent history of every evaluation.

## What to monitor

1. **Validation trend:** Every 1,000 steps, the trainer evaluates a fixed set of up to 20 validation batches and prints `sampled validation loss`. Lower is better for next-token prediction. This is a monitoring sample, not a full-validation result or a factual-accuracy score.
2. **Training versus validation:** The progress bar reports the current training batch loss. A single training batch is noisy and differs from the fixed validation sample. Sustained lower training loss alongside rising validation loss suggests overfitting; investigate rather than automatically adding epochs.
3. **Stable learning:** No non-finite loss/gradients, repeated errors, or failed saves. Verify that the latest checkpoint can be loaded and used for generation. Monitor disk space and the progress estimate.
4. **Fixed-prompt output:** Use identical prompts, seed, temperature and output length at comparison points. Review readability, relevance, repetition, gibberish, factual errors, and Afghan Dari vocabulary. Loss alone does not establish these qualities.

Keep validation observations in a small run record, because terminal progress is not a durable experiment log:

| Checkpoint / step | Training batch loss | Sampled validation loss | Output observations | Errors / decision |
| --- | --- | --- | --- | --- |
| | | | | |

## Local live dashboard

The updated trainer writes `metrics.jsonl` (history) and `status.json` (latest measurement) inside the run directory. It records training batches every 100 steps, sampled validation at checkpoint intervals, and checkpoint/pause/completion events. These local files are excluded from Git and do not require the dashboard server to be running.

If training began before the logging update, press Ctrl+C once in its terminal and wait for the safe-save message. Resume to enable logging without restarting learned weights:

```bash
.venv/bin/python -u scripts/train_epochs.py --epochs 1 --resume
```

In a separate terminal, from the repository root:

```bash
.venv/bin/python -u scripts/training_dashboard.py
```

Open `http://127.0.0.1:8765`. The server binds only to the local computer and reads run data. The page refreshes every two seconds. It shows progress, loss charts, hover inspection, series toggles, a step-range selector, rate/estimated time remaining, checkpoint position, run settings, disk space, recent events and JSON export. Token throughput is approximate (rate times batch size times sequence length). Training loss is a single batch, not an epoch average.

Stopping the dashboard does not stop training. A stale measurement is labeled; it does not prove whether the trainer process is alive. Before logging is activated, the page can display saved checkpoint progress but cannot recover loss history that existed only in the terminal. On resume, displayed chart events beyond the restored checkpoint step are discarded; the raw log remains preserved.

### Dashboard status colors

These are transparent monitoring heuristics, not a guarantee of model quality. The page always explains its reason in text, as well as color:

- Green: the latest up to four sampled-validation losses show a net decrease of at least 0.5%. With fewer than four points, this is labeled an early signal.
- Amber: insufficient history, less than 0.5% movement, an increase that is not yet sustained, a pause, unavailable live measurements, stale data (over 30 seconds), or a dashboard connection failure.
- Red: an explicit trainer error/invalid loss, or three consecutive validation increases totaling at least 2% across the last four checks. A worsening trend means review possible overfitting or instability, not that either has been proven.

Training batch loss alone never triggers a red warning. Older running trainer processes do not pick up new error-reporting code until a safe stop/resume. If they fail, the dashboard will show stale data rather than claim a known error. No automatic stopping or epoch-2 decision is performed by these colors.

## Before authorizing epoch 2

- Confirm epoch 1 actually completed; a stopped partial epoch is not a full pass.
- Record early and late validation measurements and their trend. If the trend is unclear, use a broader validation evaluation before deciding. Do not use the test split to tune the number of epochs.
- Preserve an epoch 1 checkpoint before resuming, because `last.pt` will be replaced:

```bash
cp -n outputs/wikipedia_v4_epochs/last.pt outputs/wikipedia_v4_epochs/epoch1.pt
```

- Compare fixed completions. Suggested initial prompts, to keep unchanged:
  - `کابل پایتخت`
  - `برای یادگیری زبان دری`
  - `آب در زندگی روزمره`
  - `امروز به مکتب رفتم و`
  - `The capital of Afghanistan is`

Generate each using the saved checkpoint; the current generation script seeds sampling with 123:

```bash
.venv/bin/python scripts/generate.py \
  --checkpoint outputs/wikipedia_v4_epochs/epoch1.pt \
  --prompt "کابل پایتخت" --new-tokens 80 --temperature 0.8
```

- Continue if validation prediction is still improving and output review does not expose a blocking data/training problem. If validation consistently worsens, outputs collapse into repetition, or saves/errors are unresolved, pause and investigate.
- A successful epoch does not imply fluent Dari, correct medical information, reliable facts or assistant behavior. Persian Wikipedia has missing source text and is not a dedicated Afghan Dari corpus.

Only after reviewing these results, continue toward two total epochs with:

```bash
.venv/bin/python -u scripts/train_epochs.py --epochs 2 --resume
```

Epoch 2 is a decision after evaluation, not an automatic step.
