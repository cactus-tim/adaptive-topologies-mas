"""E2 error-burst watcher.

Polls PG every 60s for failed/budget_exceeded runs in a given experiment.
Writes a structured log + ALERT banner on bursts.

Thresholds (tuned on E1 baseline 7.4% fail rate from Cerebras 429 TPM):
  - ``WARN``  : delta >= 3 fails in the last 2 min
  - ``ALERT`` : delta >= 10 fails in the last 5 min, OR
                cumulative fail rate >= 15% (3x E1 baseline)

Output: ``logs/e2_errors.log``. Each line is a single JSON event for
easy grepping. The watcher itself is read-only — it never touches runs.

Usage::

    python -m scripts.e2_error_watch --exp-id <UUID>
    python -m scripts.e2_error_watch --exp-id <UUID> --interval 60 --out logs/e2_errors.log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

_DEFAULT_LOG = Path("logs/e2_errors.log")
_WARN_DELTA = 3       # >=3 new fails in window → WARN
_WARN_WINDOW_S = 120
_ALERT_DELTA = 10     # >=10 new fails in window → ALERT
_ALERT_WINDOW_S = 300
_ALERT_RATE_PCT = 15.0  # >=15% cumulative fail rate → ALERT


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _emit(out_fh, level: str, **payload) -> None:
    record = {"at": _now(), "level": level, **payload}
    out_fh.write(json.dumps(record, default=str) + "\n")
    out_fh.flush()


async def _poll_once(engine, exp_id: str):
    async with engine.connect() as conn:
        r = (await conn.execute(text("""
            SELECT status, COUNT(*) AS n
            FROM runs
            WHERE exp_id = :e
            GROUP BY status
        """), {"e": exp_id})).mappings().all()
        counts = {row["status"]: int(row["n"]) for row in r}

        # Sample of the 5 most recent failed runs
        r = (await conn.execute(text("""
            SELECT id, topology, task_id, finish_reason,
                   LEFT(COALESCE(error, ''), 240) AS err
            FROM runs
            WHERE exp_id = :e AND status IN ('failed', 'budget_exceeded')
            ORDER BY finished_at DESC NULLS LAST
            LIMIT 5
        """), {"e": exp_id})).mappings().all()
        samples = [dict(row) for row in r]
    return counts, samples


async def run(exp_id: str, interval: int, out_path: Path) -> None:
    dsn = os.environ.get("PG_DSN") or os.environ.get("ATM_PG_DSN")
    if not dsn:
        sys.stderr.write("PG_DSN / ATM_PG_DSN not set.\n")
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("a", encoding="utf-8")

    engine = create_async_engine(dsn)

    # Sliding window of (timestamp, fails_seen) tuples.
    history: deque[tuple[float, int]] = deque(maxlen=120)  # ~2h at 60s
    last_total_fails = 0

    _emit(fh, "INFO", event="watch_start", exp_id=exp_id, interval_s=interval,
          warn=(_WARN_DELTA, _WARN_WINDOW_S),
          alert=(_ALERT_DELTA, _ALERT_WINDOW_S, _ALERT_RATE_PCT))

    try:
        while True:
            try:
                counts, samples = await _poll_once(engine, exp_id)
            except Exception as exc:
                _emit(fh, "ERROR", event="poll_failed", error=str(exc))
                await asyncio.sleep(interval)
                continue

            total = sum(counts.values())
            fails = counts.get("failed", 0) + counts.get("budget_exceeded", 0)
            completed = counts.get("completed", 0)
            running = counts.get("running", 0)

            now_ts = time.time()
            history.append((now_ts, fails))

            # Compute deltas over sliding windows
            def _delta(window_s: int) -> int:
                cutoff = now_ts - window_s
                # Find oldest sample within the window
                for ts, f in history:
                    if ts >= cutoff:
                        return fails - f
                return 0

            delta_warn = _delta(_WARN_WINDOW_S)
            delta_alert = _delta(_ALERT_WINDOW_S)
            rate_pct = (fails / total * 100.0) if total > 0 else 0.0

            if fails != last_total_fails:
                _emit(fh, "INFO", event="fail_count_change",
                      fails=fails, completed=completed, running=running,
                      delta_2m=delta_warn, delta_5m=delta_alert,
                      rate_pct=round(rate_pct, 2),
                      samples=samples)
                last_total_fails = fails

            if delta_alert >= _ALERT_DELTA:
                _emit(fh, "ALERT", event="burst_5m",
                      msg=f"{delta_alert} new fails in 5 min",
                      delta_5m=delta_alert, rate_pct=round(rate_pct, 2),
                      samples=samples)
            elif delta_warn >= _WARN_DELTA:
                _emit(fh, "WARN", event="burst_2m",
                      msg=f"{delta_warn} new fails in 2 min",
                      delta_2m=delta_warn, samples=samples)

            if rate_pct >= _ALERT_RATE_PCT and total >= 50:
                _emit(fh, "ALERT", event="rate_high",
                      msg=f"cumulative fail rate {rate_pct:.1f}% over {total} cells",
                      rate_pct=round(rate_pct, 2), fails=fails, total=total,
                      samples=samples)

            await asyncio.sleep(interval)
    finally:
        await engine.dispose()
        fh.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--interval", type=int, default=60, help="Poll interval (s)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_LOG)
    args = parser.parse_args()
    asyncio.run(run(args.exp_id, args.interval, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
