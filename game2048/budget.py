"""Cooperative budget checks. The independent supervisor enforces the hard cap."""
import time


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, used_seconds=0., reserve=30.):
        self.start = time.monotonic()
        self.seconds = float(seconds)
        self.used_before = float(used_seconds)
        self.reserve = float(reserve)

    @property
    def elapsed(self):
        return self.used_before + time.monotonic() - self.start

    @property
    def remaining(self):
        return max(0., self.seconds - self.elapsed)

    def check(self):
        if self.remaining <= self.reserve:
            raise BudgetExpired('save reserve reached; preserve the previous complete iteration')

    def can_start(self, estimate):
        return self.remaining > self.reserve + max(0., estimate)
