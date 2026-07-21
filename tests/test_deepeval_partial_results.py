import pytest

from rag_stack_evaluator.static_rag_evaluator.evaluation.metric import deepeval_metrics


class _DummyAdapter:
	def get_model_name(self):
		return "dummy-judge"

	def close(self):
		pass


class _DummyMetric:
	"""Mimics a DeepEval metric: per-case ``a_measure`` sets ``.score``.

	``_measure_batch`` drives metrics PER case under an anyio scheduler now
	(not DeepEval's ``evaluate()``), constructing one instance per case with
	``model=``/``threshold=``/``async_mode=``/``include_reason=``/``verbose_mode=``.
	"""

	def __init__(self, **kwargs):
		self.score = None
		self.error = None

	async def a_measure(self, case, _show_indicator=False):
		self.score = 1.0
		return self.score


class _ErroringMetric:
	"""Every case's ``a_measure`` raises — simulates a judge that is down."""

	def __init__(self, **kwargs):
		self.score = None
		self.error = None

	async def a_measure(self, case, _show_indicator=False):
		raise RuntimeError("judge boom")


class _PartiallyErroringMetric:
	def __init__(self, **kwargs):
		self.score = None
		self.error = None

	async def a_measure(self, case, _show_indicator=False):
		if case == "truncated":
			raise ValueError("invalid JSON after output length limit")
		self.score = 1.0
		return self.score


def test_deepeval_measure_batch_returns_per_case_scores(monkeypatch):
	"""Happy path: per-case ``a_measure`` scores are collected in input order."""
	monkeypatch.setattr(
		deepeval_metrics,
		"_build_adapter",
		lambda model, ai_client_kwargs: _DummyAdapter(),
	)

	scores = deepeval_metrics._measure_batch(
		_DummyMetric,
		metric_inputs=[object(), object(), object()],
		build_case_fn=lambda mi: mi,
		model="vllm",
		threshold=0.5,
		ai_client_kwargs={},
	)

	assert scores == [1.0, 1.0, 1.0]


def test_deepeval_all_cases_error_raises(monkeypatch):
	"""Judge-down guard: when EVERY case errors, ``_measure_batch`` RAISES rather
	than aggregating all-NaN into a silent 0.0 quality (the 2026-06-10 poisoning)."""
	monkeypatch.setattr(
		deepeval_metrics,
		"_build_adapter",
		lambda model, ai_client_kwargs: _DummyAdapter(),
	)

	with pytest.raises(RuntimeError, match="failed on 3/3 test cases"):
		deepeval_metrics._measure_batch(
			_ErroringMetric,
			metric_inputs=[object(), object(), object()],
			build_case_fn=lambda mi: mi,
			model="vllm",
			threshold=0.5,
			ai_client_kwargs={},
		)


def test_deepeval_partial_case_error_discards_batch(monkeypatch):
	"""A truncated/malformed judge case must not become a survivor-only mean.

	Diagnostic metrics enter the proposal agent's history, so even a partial
	metric is optimizer state and must trigger the controller's full-eval retry.
	"""
	monkeypatch.setattr(
		deepeval_metrics,
		"_build_adapter",
		lambda model, ai_client_kwargs: _DummyAdapter(),
	)

	with pytest.raises(
		RuntimeError,
		match="failed on 1/3 test cases.*partial metric results discarded",
	):
		deepeval_metrics._measure_batch(
			_PartiallyErroringMetric,
			metric_inputs=["ok-1", "truncated", "ok-2"],
			build_case_fn=lambda mi: mi,
			model="vllm",
			threshold=0.5,
			ai_client_kwargs={},
		)


def test_deepeval_async_config_uses_judge_concurrency(monkeypatch):
	"""Concurrency falls back to JUDGE_MAX_CONCURRENCY when no explicit override
	and no higher-priority env source is set."""
	for var in ("JUDGE_DEEPEVAL_MAX_CONCURRENT", "DEEPEVAL_MAX_CONCURRENT"):
		monkeypatch.delenv(var, raising=False)
	monkeypatch.setenv("JUDGE_MAX_CONCURRENCY", "64")

	cfg = deepeval_metrics._resolve_deepeval_async_config({})

	assert cfg.max_concurrent == 64


def test_deepeval_async_config_defaults_to_twenty(monkeypatch):
	for var in (
		"JUDGE_DEEPEVAL_MAX_CONCURRENT",
		"DEEPEVAL_MAX_CONCURRENT",
		"JUDGE_MAX_CONCURRENCY",
	):
		monkeypatch.delenv(var, raising=False)

	cfg = deepeval_metrics._resolve_deepeval_async_config({})

	assert cfg.max_concurrent == 20


def test_deepeval_async_config_explicit_override_wins(monkeypatch):
	"""An explicit ``deepeval_max_concurrent`` beats every env source."""
	monkeypatch.setenv("JUDGE_MAX_CONCURRENCY", "64")
	monkeypatch.setenv("JUDGE_DEEPEVAL_MAX_CONCURRENT", "99")

	cfg = deepeval_metrics._resolve_deepeval_async_config(
		{"deepeval_max_concurrent": 12}
	)

	assert cfg.max_concurrent == 12
