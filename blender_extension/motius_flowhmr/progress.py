# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured per-phase progress, throttled independently of inference kernels."""
import time


class InferenceProgress:
    def __init__(self, publish, clock=time.monotonic, interval=.25):
        self.publish, self.clock, self.interval = publish, clock, interval
        self.phase = None
        self.started = self.last_publish = 0.
        self.origin = 0

    def __call__(self, phase, current=0, total=0):
        now = self.clock()
        changed = phase != self.phase
        if changed:
            self.phase, self.started, self.origin = phase, now, current
        if not changed and current != total and now-self.last_publish < self.interval:
            return
        elapsed = now-self.started
        count = current-self.origin
        rate = count/elapsed if elapsed >= 3 and count > 0 else 0.
        remaining = max(0., (total-current)/rate) if rate and total else -1.
        unit = 'layers/s' if 'global layers' in phase else 'steps/s' if phase == 'Sample motion' else 'frames/s'
        self.publish('running', phase, phase=phase, current=current, total=total or 0, rate_unit=unit,
                     phase_elapsed_seconds=elapsed, rate_per_second=rate,
                     eta_seconds=remaining)
        self.last_publish = now
