"""
Evaluation Timing Utilities

Provides EvalTimer for tracking per-language, per-batch timing,
throughput metrics (tokens/sec, samples/sec), and ETA prediction.
"""

import time
from datetime import datetime, timedelta


def format_duration(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


class EvalTimer:
    """Tracks evaluation timing with per-language breakdown and ETA prediction.

    Usage:
        timer = EvalTimer(total_languages=11)

        for lang in languages:
            timer.start_language(lang)
            # ... inference ...
            timer.record_batch(tokens=12345, samples=250, duration=120.5)
            timer.end_language()

        timing_stats = timer.get_stats()
    """

    def __init__(self, total_languages, total_runs=1):
        self.total_languages = total_languages
        self.total_runs = total_runs
        self.global_start = time.time()

        # Per-language tracking
        self.language_times = []  # list of dicts
        self._lang_start = None
        self._lang_name = None
        self._lang_batches = []

        # Per-run tracking (for xreasoning multi-run)
        self.run_times = []  # list of dicts
        self._run_start = None
        self._run_idx = None
        self._completed_runs = 0

    # ── Language-level ──────────────────────────────────────────

    def start_language(self, lang_name):
        self._lang_start = time.time()
        self._lang_name = lang_name
        self._lang_batches = []

    def record_batch(self, tokens, samples, duration):
        """Record a completed batch within the current language."""
        self._lang_batches.append({
            "tokens": tokens,
            "samples": samples,
            "duration_sec": duration,
        })

    def end_language(self, skipped=False):
        """End timing for current language and print progress + ETA."""
        duration = time.time() - self._lang_start
        total_tokens = sum(b["tokens"] for b in self._lang_batches)
        total_samples = sum(b["samples"] for b in self._lang_batches)

        entry = {
            "language": self._lang_name,
            "duration_sec": duration,
            "tokens_generated": total_tokens,
            "samples_processed": total_samples,
            "tokens_per_sec": total_tokens / duration if duration > 0 else 0,
            "samples_per_sec": total_samples / duration if duration > 0 else 0,
            "batches": self._lang_batches,
            "skipped": skipped,
        }
        self.language_times.append(entry)
        self._print_progress(entry)
        return entry

    def _print_progress(self, entry):
        completed = len(self.language_times)
        elapsed = time.time() - self.global_start
        lang = entry["language"]
        dur = entry["duration_sec"]

        if entry["skipped"]:
            print(f"\n[Timer] {lang}: skipped (loaded from cache) [{format_duration(dur)}]")
        else:
            tps = entry["tokens_per_sec"]
            sps = entry["samples_per_sec"]
            print(f"\n[Timer] {lang}: {format_duration(dur)} | "
                  f"{entry['tokens_generated']} tokens ({tps:.1f} tok/s) | "
                  f"{entry['samples_processed']} samples ({sps:.2f} sam/s)")

        # ETA based on non-skipped languages only
        active_times = [t for t in self.language_times if not t["skipped"]]
        remaining_langs = self.total_languages - completed

        # Account for multi-run: scale by remaining runs
        remaining_in_current_run = remaining_langs
        remaining_future_runs = max(0, self.total_runs - self._completed_runs - 1)

        if active_times and (remaining_in_current_run > 0 or remaining_future_runs > 0):
            avg_per_lang = sum(t["duration_sec"] for t in active_times) / len(active_times)
            eta_current_run = avg_per_lang * remaining_in_current_run
            eta_future_runs = avg_per_lang * self.total_languages * remaining_future_runs
            eta_total = eta_current_run + eta_future_runs

            eta_time = datetime.now() + timedelta(seconds=eta_total)
            print(f"[Timer] Progress: {completed}/{self.total_languages} languages | "
                  f"Elapsed: {format_duration(elapsed)} | "
                  f"ETA: {format_duration(eta_total)} (finish ~{eta_time.strftime('%H:%M:%S')})")

            if active_times:
                avg_tps = sum(t["tokens_per_sec"] for t in active_times) / len(active_times)
                print(f"[Timer] Avg throughput: {avg_tps:.1f} tok/s | "
                      f"Avg per language: {format_duration(avg_per_lang)}")
        else:
            print(f"[Timer] Progress: {completed}/{self.total_languages} languages | "
                  f"Elapsed: {format_duration(elapsed)}")

    # ── Run-level (for xreasoning multi-run) ────────────────────

    def start_run(self, run_idx):
        self._run_start = time.time()
        self._run_idx = run_idx
        # Reset per-language tracking for this run
        self.language_times = []

    def end_run(self, skipped=False):
        duration = time.time() - self._run_start
        run_entry = {
            "run_idx": self._run_idx,
            "duration_sec": duration,
            "language_details": list(self.language_times),
            "skipped": skipped,
        }
        self.run_times.append(run_entry)
        self._completed_runs += 1

        active_runs = [r for r in self.run_times if not r["skipped"]]
        remaining_runs = self.total_runs - len(self.run_times)

        if active_runs and remaining_runs > 0:
            avg_per_run = sum(r["duration_sec"] for r in active_runs) / len(active_runs)
            eta = avg_per_run * remaining_runs
            eta_time = datetime.now() + timedelta(seconds=eta)
            elapsed = time.time() - self.global_start
            print(f"\n[Timer] Run {self._run_idx + 1} done: {format_duration(duration)} | "
                  f"Runs: {len(self.run_times)}/{self.total_runs} | "
                  f"Elapsed: {format_duration(elapsed)} | "
                  f"ETA: {format_duration(eta)} (finish ~{eta_time.strftime('%H:%M:%S')})")
        else:
            elapsed = time.time() - self.global_start
            print(f"\n[Timer] Run {self._run_idx + 1} done: {format_duration(duration)} | "
                  f"Runs: {len(self.run_times)}/{self.total_runs} | "
                  f"Elapsed: {format_duration(elapsed)}")

        return run_entry

    # ── Statistics ──────────────────────────────────────────────

    def get_stats(self):
        """Return timing statistics dict for saving to results JSON."""
        elapsed = time.time() - self.global_start

        # Gather all language timing across runs
        all_lang_times = []
        if self.run_times:
            for run in self.run_times:
                all_lang_times.extend(run["language_details"])
        else:
            all_lang_times = list(self.language_times)

        active_lang_times = [t for t in all_lang_times if not t["skipped"]]

        per_language_timing = {}
        for t in all_lang_times:
            lang = t["language"]
            if lang not in per_language_timing:
                per_language_timing[lang] = []
            per_language_timing[lang].append({
                "duration_sec": round(t["duration_sec"], 2),
                "tokens_generated": t["tokens_generated"],
                "tokens_per_sec": round(t["tokens_per_sec"], 1),
                "samples_processed": t["samples_processed"],
                "skipped": t["skipped"],
            })

        stats = {
            "total_elapsed_sec": round(elapsed, 2),
            "total_elapsed_hours": round(elapsed / 3600, 4),
            "per_language_timing": per_language_timing,
        }

        if active_lang_times:
            total_tokens = sum(t["tokens_generated"] for t in active_lang_times)
            total_inference_sec = sum(t["duration_sec"] for t in active_lang_times)
            stats["total_tokens_generated"] = total_tokens
            stats["total_inference_sec"] = round(total_inference_sec, 2)
            stats["avg_tokens_per_sec"] = round(total_tokens / total_inference_sec, 1) if total_inference_sec > 0 else 0
            stats["avg_sec_per_language"] = round(total_inference_sec / len(active_lang_times), 2)

        if self.run_times:
            stats["per_run_timing"] = []
            for run in self.run_times:
                stats["per_run_timing"].append({
                    "run_idx": run["run_idx"],
                    "duration_sec": round(run["duration_sec"], 2),
                    "skipped": run["skipped"],
                })

        return stats
