"""Learning rate over training time: linear warmup, then cosine decay."""
import math


def learning_rate_at(step, total_steps, base_lr, warmup=0, min_ratio=1.0):
    """Return the learning rate for a 0-based optimizer step.

    The defaults (no warmup, min_ratio 1.0) hold the rate constant, which is
    what runs started before this schedule existed used.
    """
    if total_steps < 1 or base_lr <= 0 or not 0.0 <= min_ratio <= 1.0 or warmup < 0:
        raise ValueError("Need positive totals, a base rate and a ratio within 0..1")
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    if min_ratio == 1.0:
        return base_lr
    progress = min((step - warmup) / max(total_steps - warmup, 1), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
