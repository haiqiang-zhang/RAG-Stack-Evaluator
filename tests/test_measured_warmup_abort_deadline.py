"""CPU-only checks of the real warmup timer and gate code, with virtual time.

AST loading executes the source functions without importing package initializers,
GPU libraries, HTTP clients, or runtime probes. Only clocks, task scheduling, and
the final start/failure sinks are substituted; support and stability gates are
compiled unchanged from serving_runtime.py.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / "rag_stack_evaluator/static_rag_evaluator/measured/serving_runtime.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = {
    node.name: node for node in ast.walk(TREE)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}


def load_functions(names, namespace):
    module = ast.Module(body=[FUNCTIONS[name] for name in names], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec", flags=0x1000000), namespace)


class TimerHarness:
    """Run the production closures with controlled scheduling, no wall wait."""

    def __init__(self, deadline=900.0):
        self.clock = 1000.0
        self.sleeps = []
        self.starts = []
        self.failures = []
        self.gate_calls = []
        self.on_sleep = None
        self.ns = {
            "math": math,
            "time": SimpleNamespace(perf_counter=lambda: self.clock),
            "asyncio": SimpleNamespace(sleep=self.sleep),
            "logger": logging.getLogger(__name__),
            "state_lock": asyncio.Lock(),
            "warmup_cap": 240.0,
            "warmup_abort_deadline": deadline,
            "warmup_start": 1000.0,
            "warmup_completed": 100,
            "measurement_started": False,
            "measurement_end": None,
            "warmup_done_ts": [float(i) for i in range(100)],
            "workload_support_complete_warmup_completed": None,
            "saturation_stability_proof_start_completion": None,
            "saturation_stability_span": 10,
            "adapter_done": True,
            "latest_population_ramp": SimpleNamespace(
                complete_event=SimpleNamespace(is_set=lambda: True),
                cancelled_before_activation=0,
            ),
            "active_driver_count": 2,
            "driver_tasks": [object(), object()],
            "_WARMUP_RATE_STABILITY_RELATIVE_TOLERANCE": 0.10,
            "self": SimpleNamespace(warmup_queries=128, _saturation={
                "saturated": True,
                "candidate_identity": "engine_backlog:generator",
                "evidence": "engine_backlog:generator",
            }),
            "warmup_original_deadline_probe": {
                "scheduled_offset_s": 240.0,
                "observed_offset_s": None,
                "measurement_already_started": False,
                "gate_checked": False,
                "gate_ready": None,
                "abort_deferred": False,
            },
            "start_measurement_locked": self.start,
            "fail_warmup_locked": self.failures.append,
        }
        load_functions([
            "_completion_rate_stable", "_rate_stable", "_warmup_rate_stable",
            "_warmup_gate_ready", "_maybe_start_after_warmup_locked", "warmup_timer",
        ], self.ns)
        self.real_gate = self.ns["_maybe_start_after_warmup_locked"]

        def observed_gate(now):
            self.gate_calls.append(now - 1000.0)
            return self.real_gate(now)

        self.ns["_maybe_start_after_warmup_locked"] = observed_gate

    async def sleep(self, delay):
        self.sleeps.append(delay)
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleeps), delay)
        self.clock += delay

    def start(self, now, reason, gate):
        self.starts.append((now - 1000.0, reason, gate))
        self.ns["measurement_started"] = True

    def support_with_fresh_proof(self, count=20):
        self.ns["workload_support_complete_warmup_completed"] = 100
        self.ns["saturation_stability_proof_start_completion"] = 100
        self.ns["warmup_done_ts"] = [float(i) for i in range(100 + count)]
        self.ns["warmup_completed"] = 100 + count

    def run(self):
        asyncio.run(self.ns["warmup_timer"]())


class AbortDeadlineTests(unittest.TestCase):
    def deadline(self, raw, reference=240.0):
        namespace = {"math": math, "os": os}
        load_functions(["_warmup_abort_deadline_s"], namespace)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_STACK_WARMUP_ABORT_DEADLINE_S", None)
            if raw is not None:
                os.environ["RAG_STACK_WARMUP_ABORT_DEADLINE_S"] = raw
            return namespace["_warmup_abort_deadline_s"](reference)

    def test_default_and_explicit_reference_preserve_original_deadline(self):
        self.assertEqual(self.deadline(None), 240.0)
        self.assertEqual(self.deadline("240"), 240.0)
        self.assertEqual(self.deadline("900"), 900.0)

    def test_invalid_or_reduced_deadlines_fail_closed(self):
        for value in ("239.99", "-1", "0", "nan", "inf", "-inf", "bad", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.deadline(value)
        for reference in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.deadline(None, reference)

    def test_default_unready_timer_rejects_at_original_reference(self):
        h = TimerHarness(deadline=240.0)
        h.run()
        self.assertEqual(h.sleeps, [240.0])
        self.assertEqual(h.gate_calls, [240.0])
        self.assertEqual(len(h.failures), 1)
        self.assertIn("cap=240.000s", h.failures[0])
        self.assertFalse(h.ns["warmup_original_deadline_probe"]["abort_deferred"])

    def test_extended_unready_timer_keeps_original_probe_then_fails_at_900(self):
        h = TimerHarness()
        h.run()
        self.assertEqual(h.sleeps, [240.0, 660.0])
        self.assertEqual(h.gate_calls, [240.0, 900.0])
        self.assertEqual(h.starts, [])
        self.assertEqual(len(h.failures), 1)
        self.assertIn("cap=900.000s", h.failures[0])
        self.assertIn("workload_support_complete=False", h.failures[0])
        self.assertEqual(h.ns["warmup_original_deadline_probe"], {
            "scheduled_offset_s": 240.0,
            "observed_offset_s": 240.0,
            "measurement_already_started": False,
            "gate_checked": True,
            "gate_ready": False,
            "abort_deferred": True,
        })

    def test_ready_original_probe_opens_equally_with_or_without_extension(self):
        outcomes = []
        for deadline in (240.0, 900.0):
            h = TimerHarness(deadline)
            h.support_with_fresh_proof()
            h.run()
            self.assertEqual(h.sleeps, [240.0])
            self.assertEqual(h.failures, [])
            self.assertTrue(h.ns["warmup_original_deadline_probe"]["gate_ready"])
            outcomes.append((h.starts, h.gate_calls))
        self.assertEqual(*outcomes)

    def test_completion_before_reference_remains_identical(self):
        for deadline in (240.0, 900.0):
            h = TimerHarness(deadline)
            h.support_with_fresh_proof()
            self.assertTrue(h.real_gate(1120.0))
            h.run()
            self.assertEqual(h.starts[0][0], 120.0)
            self.assertEqual(h.gate_calls, [])
            self.assertEqual(h.failures, [])
            probe = h.ns["warmup_original_deadline_probe"]
            self.assertTrue(probe["measurement_already_started"])
            self.assertFalse(probe["gate_checked"])

    def test_delayed_original_probe_retains_gate_before_abort_ordering(self):
        h = TimerHarness()
        h.support_with_fresh_proof()
        h.on_sleep = lambda _index, _delay: setattr(h, "clock", h.clock + 2.0)
        h.run()
        self.assertEqual(h.starts[0][0], 242.0)
        self.assertEqual(h.ns["warmup_original_deadline_probe"]["observed_offset_s"], 242.0)
        self.assertFalse(h.ns["warmup_original_deadline_probe"]["abort_deferred"])
        self.assertEqual(h.failures, [])

    def test_post_reference_completion_still_needs_support_and_fresh_proof(self):
        h = TimerHarness()

        def complete_after_reference(index, _delay):
            if index != 2:
                return
            self.assertFalse(h.real_gate(1300.0))  # Missing workload support.
            h.support_with_fresh_proof(count=19)
            self.assertFalse(h.real_gate(1320.0))  # Only 19 of 20 fresh completions.
            h.support_with_fresh_proof(count=20)
            self.assertTrue(h.real_gate(1330.0))

        h.on_sleep = complete_after_reference
        h.run()
        self.assertEqual(h.starts[0][0], 330.0)
        self.assertEqual(h.starts[0][2], "warmup_completion_rate_stable")
        self.assertEqual(h.failures, [])
        self.assertTrue(h.ns["warmup_original_deadline_probe"]["abort_deferred"])

    def test_post_reference_still_requires_backlog_and_activated_population(self):
        for missing in ("backlog", "adapter", "ramp", "population"):
            with self.subTest(missing=missing):
                h = TimerHarness()
                h.support_with_fresh_proof()
                if missing == "backlog":
                    h.ns["self"]._saturation["saturated"] = False
                elif missing == "adapter":
                    h.ns["adapter_done"] = False
                elif missing == "ramp":
                    h.ns["latest_population_ramp"].cancelled_before_activation = 1
                else:
                    h.ns["active_driver_count"] = 1
                h.run()
                self.assertEqual(h.starts, [])
                self.assertEqual(len(h.failures), 1)

    def test_final_deadline_ready_probe_preserves_gate_before_abort(self):
        h = TimerHarness()
        h.on_sleep = lambda index, _delay: h.support_with_fresh_proof() if index == 2 else None
        h.run()
        self.assertEqual(h.starts[0][0], 900.0)
        self.assertEqual(h.failures, [])

    def test_no_warmup_skips_both_timers(self):
        h = TimerHarness()
        h.ns["self"].warmup_queries = 0
        h.run()
        self.assertEqual(h.sleeps, [])
        self.assertEqual(h.gate_calls, [])

    def test_defensive_completion_timeout_uses_only_abort_deadline(self):
        driver = FUNCTIONS["driver"]
        branches = [node for node in ast.walk(driver) if isinstance(node, ast.If)
                    and "state.done_ts - warmup_start >= warmup_abort_deadline" == ast.unparse(node.test)]
        self.assertEqual(len(branches), 1)
        code = compile(ast.Module(body=branches, type_ignores=[]), str(SOURCE), "exec")
        h = TimerHarness()
        for elapsed in (240.0, 899.0):
            h.ns["state"] = SimpleNamespace(done_ts=1000.0 + elapsed)
            exec(code, h.ns)
            self.assertEqual(h.failures, [])
        h.ns["state"] = SimpleNamespace(done_ts=1900.0)
        exec(code, h.ns)
        self.assertEqual(len(h.failures), 1)
        self.assertIn("cap=900.000s", h.failures[0])

    def test_reference_policy_inputs_are_not_replaced_by_abort_deadline(self):
        # These formulas are protocol inputs, not abort decisions. Compile their
        # real expressions to ensure the new operational knob cannot affect them.
        parent = FUNCTIONS["_run_closed_loop_saturated"]
        expressions = {ast.unparse(node): node for node in ast.walk(parent)
                       if isinstance(node, ast.BinOp)}
        half = expressions["warmup_cap / 2.0"]
        quarter = expressions["warmup_cap * 0.25"]
        ns = {"warmup_cap": 240.0, "warmup_abort_deadline": 900.0}
        self.assertEqual(eval(compile(ast.Expression(half), str(SOURCE), "eval"), ns), 120.0)
        self.assertEqual(min(10.0, eval(compile(ast.Expression(quarter), str(SOURCE), "eval"), ns)), 10.0)
        self.assertNotIn("warmup_abort_deadline", ast.unparse(FUNCTIONS["measured_timer"]))
        self.assertNotIn("warmup_abort_deadline", ast.unparse(FUNCTIONS["population_adapter"]))


if __name__ == "__main__":
    unittest.main()
