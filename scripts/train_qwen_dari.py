# /// script
# dependencies = ["trl>=1.0", "peft>=0.7", "transformers>=4.45", "datasets>=3", "trackio"]
# ///
"""LoRA fine-tune an open model on Afghan curriculum Dari text.

Runs unchanged locally, on a RunPod pod, or as a Hugging Face job.
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/dari_finetune_v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--dataset', default=str(DATA), help='Local folder with train/validation jsonl, or a Hub id')
    p.add_argument('--output', default='outputs/qwen-dari')
    p.add_argument('--hub-model-id', help='Push the adapter here; the job environment is ephemeral')
    p.add_argument('--epochs', type=float, default=3.0)
    p.add_argument('--max-steps', type=int, default=-1, help='Overrides epochs; use a few for a smoke test')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lora-r', type=int, default=32)
    p.add_argument('--trackio', action='store_true', help='Report metrics to Trackio')
    args = p.parse_args()

    source = Path(args.dataset)
    if source.exists():
        data = load_dataset('json', data_files={'train': str(source / 'train.jsonl'),
                                                'validation': str(source / 'validation.jsonl')})
    else:
        data = load_dataset(args.dataset)
    print({split: len(rows) for split, rows in data.items()}, flush=True)

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        logging_steps=5,
        eval_strategy='steps',
        eval_steps=25,
        save_strategy='steps',
        save_steps=50,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        report_to='trackio' if args.trackio else 'none',
        run_name='qwen-dari-curriculum',
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id,
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=data['train'],
        eval_dataset=data['validation'],
        # LoRA keeps a 4B model trainable on one 24 GB card.
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                               task_type='CAUSAL_LM'),
    )
    trainer.train()
    if args.hub_model_id:
        trainer.push_to_hub()
    print('Done. Adapter saved to', args.output)


if __name__ == '__main__':
    main()
