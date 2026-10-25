"""Spend guard for paid training. Stops the run before the money runs out."""
import json
import time
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    """Tracks billed hours against a hard ceiling, surviving resume.

    Accounting is wall-clock against a stated hourly rate: it is an estimate of
    the provider's bill, not the bill itself. Always set the ceiling below what
    you are willing to lose.
    """

    def __init__(self, folder, limit_usd, rate_per_hour, max_failures=3):
        if limit_usd <= 0 or rate_per_hour <= 0:
            raise ValueError('Budget limit and hourly rate must both be positive')
        self.path = Path(folder) / 'budget.json'
        self.limit_usd = float(limit_usd)
        self.rate = float(rate_per_hour)
        self.max_failures = max_failures
        self.started = time.time()
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        # Hours billed by earlier sessions of this run carry forward.
        self.previous_hours = float(state.get('billed_hours', 0.0))
        self.failures = int(state.get('consecutive_failures', 0))

    @property
    def billed_hours(self):
        return self.previous_hours + (time.time() - self.started) / 3600

    @property
    def spent_usd(self):
        return self.billed_hours * self.rate

    @property
    def remaining_usd(self):
        return self.limit_usd - self.spent_usd

    def record_failure(self):
        self.failures += 1
        self.save()
        return self.failures

    def record_success(self):
        self.failures = 0

    def check(self):
        """Raise when the run must stop. Callers save a checkpoint first."""
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f'Budget exhausted: ${self.spent_usd:.2f} of ${self.limit_usd:.2f} '
                f'after {self.billed_hours:.2f} billed hours')
        if self.failures >= self.max_failures:
            raise BudgetExceeded(f'Stopping after {self.failures} consecutive failures')
        return self.remaining_usd

    def save(self):
        self.path.write_text(json.dumps(dict(
            billed_hours=round(self.billed_hours, 4),
            spent_usd=round(self.spent_usd, 4),
            limit_usd=self.limit_usd,
            rate_per_hour=self.rate,
            remaining_usd=round(self.remaining_usd, 4),
            consecutive_failures=self.failures,
            updated=time.strftime('%Y-%m-%d %H:%M:%S')), indent=2) + '\n', encoding='utf-8')
