"""Sample tokens from AMC's checkpoint. Output will be nonsense after toy training."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.tokenization import get_tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt, new_tokens, temperature):
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        raise ValueError("Prompt cannot be empty")
    for _ in range(new_tokens):
        context = torch.tensor([input_ids[-model.cfg.context_length:]], dtype=torch.long)
        logits, _ = model(context)
        distribution = torch.softmax(logits[0, -1] / temperature, dim=-1)
        index = torch.multinomial(distribution, 1).item()
        if index == tokenizer.eos_id:
            break
        input_ids.append(index)
    return tokenizer.decode(input_ids)


def terminal_text(text):
    # Random byte predictions can contain terminal control characters.
    return "".join(c if c.isprintable() or c == "\n" else repr(c)[1:-1] for c in text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", default="سلام")
    p.add_argument("--new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--interactive", action="store_true", help="Keep accepting prompts in the terminal")
    args = p.parse_args()
    if args.new_tokens < 1 or args.temperature <= 0:
        p.error("--new-tokens and --temperature must be positive")
    torch.set_num_threads(2)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = get_tokenizer(ckpt["tokenizer_kind"], ckpt.get("tokenizer_path"))
    model = AMCTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    torch.manual_seed(123)
    if not args.interactive:
        if not args.prompt:
            p.error("Prompt cannot be empty")
        print(terminal_text(generate(model, tokenizer, args.prompt, args.new_tokens, args.temperature)))
        return
    print(f"AMC loaded: {model.count_parameters():,} parameters, {ckpt.get('steps', 'unknown')} training steps.")
    print("Early text-completion experiment: output is mostly gibberish. Each prompt is independent.")
    print("Type a prompt and press Enter. Type /quit or press Ctrl+C to exit.\n")
    while True:
        try:
            prompt = input("You > ")
            if prompt.strip() == "/quit":
                break
            if not prompt.strip():
                continue
            print("AMC completion >", terminal_text(generate(model, tokenizer, prompt, args.new_tokens, args.temperature)), "\n")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
