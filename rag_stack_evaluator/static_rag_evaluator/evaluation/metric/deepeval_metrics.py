import asyncio
import logging
import os
from typing import Any, List, Optional

import anyio
from deepeval import evaluate
from deepeval.evaluate.configs import (
	AsyncConfig,
	CacheConfig,
	DisplayConfig,
	ErrorConfig,
)
from deepeval.metrics import (
	ContextualPrecisionMetric,
	ContextualRecallMetric,
	ContextualRelevancyMetric,
	AnswerRelevancyMetric,
	FaithfulnessMetric,
	GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class _AnswerCorrectnessMetric(GEval):
	"""GEval-based factual correctness metric.

	Custom subclass so the rest of the codebase can treat it like any other
	DeepEval metric class (init takes only ``model`` / ``threshold`` /
	``async_mode``, same as ``FaithfulnessMetric`` etc.). Criteria and
	evaluation_params are fixed here.

	Compares ``actual_output`` (LLM-generated answer) to ``expected_output``
	(``generation_gt`` from the qa.parquet) and scores factual correctness on
	[0, 1]. Penalizes hallucination, missing key facts, and contradictions —
	complements ``FaithfulnessMetric`` (which checks claims against retrieval
	context, NOT against ground-truth answer).
	"""

	def __init__(self, **kwargs):
		# GEval.__init__ doesn't accept include_reason / verbose_mode that
		# the standard DeepEval metric callers pass. DeepEval's evaluate()
		# copies metrics by replaying constructor args from vars(metric), so
		# also drop fixed GEval fields that this subclass owns.
		kwargs.pop("include_reason", None)
		kwargs.pop("verbose_mode", None)
		kwargs.pop("name", None)
		kwargs.pop("criteria", None)
		kwargs.pop("evaluation_params", None)
		super().__init__(
			name="Answer Correctness",
			criteria=(
				"Determine whether the actual output is factually correct "
				"based on the expected output. Penalize hallucination, "
				"missing key facts, and contradictions with the expected "
				"output."
			),
			evaluation_params=[
				LLMTestCaseParams.ACTUAL_OUTPUT,
				LLMTestCaseParams.EXPECTED_OUTPUT,
			],
			**kwargs,
		)

from rag_stack.ai_clients import get_ai_client
from rag_stack.ai_clients.base import AIClient
from rag_stack_evaluator.static_rag_evaluator.evaluation.metric.util import autorag_metric_loop
from rag_stack_evaluator.static_rag_evaluator.schema.metricinput import MetricInput

logger = logging.getLogger("RAG-Stack")


class _AIClientDeepEvalAdapter(DeepEvalBaseLLM):
	"""Adapter: exposes any rag_stack ``AIClient`` as a ``DeepEvalBaseLLM``.

	DeepEval metrics call ``.a_generate(prompt)`` (when ``async_mode=True``)
	or ``.generate(prompt)`` (sync). We translate both into a single-turn
	``AIClient.chat([{"role":"user","content": prompt}])`` call.

	The concrete client type is decided by the YAML config's ``model`` field,
	which is passed to ``get_ai_client(model_name, **kwargs)``:

	    model: openai/gpt-4o-mini        → OpenAIClient
	    model: openai/codex-cli/gpt-5.4  → CodexCliClient (wraps `codex exec`)
	    model: anthropic/claude-code/sonnet → ClaudeCodeClient
	    model: anthropic/minimax/MiniMax-M2.7 → AnthropicClient (MiniMax)
	    model: anthropic/claude-sonnet-4-6   → AnthropicClient (real Anthropic)
	"""

	def __init__(self, client: AIClient):
		self._client = client
		# DeepEval sometimes walks .model; point it at ourselves to no-op.
		self.model = self
		self.name = client.model

	def load_model(self):
		return self

	def generate(self, prompt: str, schema=None, **kwargs):
		"""Sync wrapper — run the async path on a throwaway loop.

		When DeepEval passes ``schema=<pydantic class>`` (via
		``a_generate_with_schema``), route to ``structured_output`` so the
		underlying client can enforce the JSON schema (OpenAI structured
		outputs, Codex ``--output-schema``, Claude Code JSON mode) and
		return a validated pydantic instance directly. That skips DeepEval's
		``trimAndLoadJson`` regex path which is fragile for Codex stdout
		(reasoning prose wraps the JSON).
		"""
		messages = [{"role": "user", "content": prompt}]
		coro = (
			self._client.structured_output(messages, response_format=schema)
			if schema is not None
			else self._client.chat(messages)
		)
		if _in_running_loop():
			return anyio.from_thread.run(lambda: coro)
		return asyncio.run(coro)

	async def a_generate(self, prompt: str, schema=None, **kwargs):
		messages = [{"role": "user", "content": prompt}]
		if schema is not None:
			return await self._client.structured_output(
				messages, response_format=schema
			)
		return await self._client.chat(messages)

	def get_model_name(self) -> str:
		return self._client.model

	def close(self):
		"""Best-effort cleanup of any long-lived handles the client may hold."""
		client = self._client
		# OpenAIClient holds an AsyncClient with an httpx connection pool
		inner = getattr(client, "_client", None)
		if inner is not None and hasattr(inner, "_client"):
			try:
				inner._client = None
			except Exception:
				pass


def _in_running_loop() -> bool:
	try:
		asyncio.get_running_loop()
		return True
	except RuntimeError:
		return False


def _build_test_case(
	input: Optional[str] = None,
	actual_output: Optional[str] = None,
	expected_output: Optional[str] = None,
	retrieval_context: Optional[List[str]] = None,
) -> LLMTestCase:
	return LLMTestCase(
		input=input or "",
		actual_output=actual_output,
		expected_output=expected_output,
		retrieval_context=retrieval_context,
	)


def _build_adapter(model: str, ai_client_kwargs: dict) -> _AIClientDeepEvalAdapter:
	"""Instantiate an AIClient from the YAML config and wrap it for DeepEval."""
	client = get_ai_client(model, **ai_client_kwargs)
	return _AIClientDeepEvalAdapter(client)


def _env_float(name: str, default: float) -> float:
	value = os.environ.get(name)
	if value is None or value == "":
		return default
	try:
		return float(value)
	except ValueError as e:
		raise ValueError(f"{name} must be a number, got {value!r}") from e


def _resolve_deepeval_async_config(ai_client_kwargs: dict) -> AsyncConfig:
	max_concurrent = ai_client_kwargs.pop("deepeval_max_concurrent", None)
	if max_concurrent is None:
		max_concurrent = (
			os.environ.get("JUDGE_DEEPEVAL_MAX_CONCURRENT")
			or os.environ.get("DEEPEVAL_MAX_CONCURRENT")
			or ai_client_kwargs.get("max_concurrency")
			or os.environ.get("JUDGE_MAX_CONCURRENCY")
			or 20
		)

	throttle_value = ai_client_kwargs.pop("deepeval_throttle_value", None)
	if throttle_value is None:
		throttle_value = _env_float("JUDGE_DEEPEVAL_THROTTLE_VALUE", 0.0)

	return AsyncConfig(
		max_concurrent=int(max_concurrent),
		throttle_value=float(throttle_value),
	)


def _measure_batch(
	metric_class,
	metric_inputs,
	build_case_fn,
	model: str,
	threshold: float,
	ai_client_kwargs: dict,
):
	"""Evaluate each test case using DeepEval's native evaluator.

	DeepEval owns async scheduling, default concurrency, deadlines,
	missing-param handling, and metric result materialization. rag-stack only
	adapts its AIClient and extracts the per-case scores for the existing
	aggregation path.
	"""
	ai_client_kwargs = dict(ai_client_kwargs)
	# Legacy YAMLs may still carry these old rag-stack-side scheduling knobs.
	# Drop them so DeepEval's own defaults apply and provider clients do not see
	# unknown keyword arguments.
	ai_client_kwargs.pop("batch_size", None)
	ai_client_kwargs.pop("throttle_value", None)
	async_config = _resolve_deepeval_async_config(ai_client_kwargs)
	# The AIClient has its OWN CapacityLimiter (default 10) — a second bound below
	# our per-case scheduler. Lift it to the scheduler concurrency so the two agree
	# (else 40-way scheduling throttles to 10 at the client). Caller-set higher wins.
	ai_client_kwargs["max_concurrency"] = max(
		int(ai_client_kwargs.get("max_concurrency", 0) or 0),
		int(async_config.max_concurrent),
	)
	adapter = _build_adapter(model, ai_client_kwargs)
	test_cases = [build_case_fn(mi) for mi in metric_inputs]
	logger.info(
		f"[judge] {metric_class.__name__}: start "
		f"cases={len(test_cases)} model={adapter.get_model_name()} "
		f"deepeval_max_concurrent={async_config.max_concurrent}"
	)

	try:
		# Drive the DeepEval metric PER CASE under our own bounded-concurrency
		# scheduler instead of DeepEval's evaluate(). evaluate() runs an
		# asyncio.gather with a GLOBAL deadline that CANCELS still-pending judge
		# tasks under load — the "incomplete result batch" failure that forced
		# concurrency down to 8. Per-case a_measure under an anyio CapacityLimiter
		# has NO global deadline: each case stands alone, the AIClient retries
		# transient blips, so concurrency can run 30-50 (deepseek sustains 50/50
		# fine; the cap was DeepEval's scheduler, not the API). One metric instance
		# PER case — a_measure mutates ``.score`` so a single instance can't be
		# shared across concurrent cases. Missing-param cases SKIP (score None, not
		# an error), matching the old skip_on_missing_params behaviour.
		import anyio
		conc = max(1, int(async_config.max_concurrent))
		scores: list[float | None] = [None] * len(test_cases)
		_errs: list[str | None] = [None] * len(test_cases)
		# GEval regenerates its evaluation_steps PER INSTANCE, and one instance
		# is built per case — one extra judge call per case (~100/eval for the
		# 100-query correctness metric). a_measure caches the generated steps on
		# the instance, so the first measured case's steps are harvested and
		# injected into every later instance (GEval short-circuits when steps
		# are provided). Judge behaviour is identical — the steps are the same
		# ones every instance would have generated from the fixed criteria.
		_geval = isinstance(metric_class, type) and issubclass(metric_class, GEval)
		geval_steps: list | None = None

		async def _measure_one(idx, case, _limiter):
			nonlocal geval_steps
			async with _limiter:
				kw = dict(
					model=adapter, threshold=threshold, async_mode=True,
					include_reason=False, verbose_mode=False,
				)
				if _geval and geval_steps:
					kw["evaluation_steps"] = list(geval_steps)
				m = metric_class(**kw)
				try:
					await m.a_measure(case, _show_indicator=False)
					scores[idx] = m.score
					if _geval and geval_steps is None and getattr(m, "evaluation_steps", None):
						geval_steps = list(m.evaluation_steps)
					err = getattr(m, "error", None)
					if err:
						_errs[idx] = str(err)
				except Exception as e:  # noqa: BLE001 — isolate per-case failure
					if "MissingTestCaseParams" in type(e).__name__:
						pass  # skip: score stays None, not counted as an error
					else:
						_errs[idx] = f"{type(e).__name__}: {e}"

		async def _run_all():
			try:
				limiter = anyio.CapacityLimiter(conc)
				cases = list(enumerate(test_cases))
				if _geval and cases:
					# Steps-priming case runs alone (its failure just falls back to
					# per-case generation for the rest — the old behaviour).
					i0, c0 = cases[0]
					await _measure_one(i0, c0, limiter)
					cases = cases[1:]
				async with anyio.create_task_group() as tg:
					for i, c in cases:
						tg.start_soon(_measure_one, i, c, limiter)
			finally:
				# Close the judge's HTTP pool ON THIS loop. The adapter's sync
				# close() below only drops the reference; the orphaned
				# AsyncClient then schedules its own aclose() at GC time onto
				# the loop anyio.run() has already closed — every metric batch
				# leaked one unretrieved "RuntimeError: Event loop is closed"
				# task exception (07-08).
				try:
					await adapter._client.aclose()
				except Exception as _e:  # noqa: BLE001 — cleanup must not mask results
					logger.warning(f"[judge] async client close failed: {_e}")

		anyio.run(_run_all)
		errors: list[str] = [e for e in _errs if e]
		# Judge-integrity guard: ANY real per-case error invalidates the metric
		# batch.  Averaging only the surviving cases is not an unbiased estimate:
		# output-length failures, for example, disproportionately remove the
		# high-top-k cases.  Worse, diagnostic sub-metrics are fed to proposal
		# agents even when they are not part of the numeric objective.  Raise so
		# the controller retries the complete evaluation and, if retries remain
		# unsuccessful, excludes it from optimizer state.  Intentional
		# MissingTestCaseParams skips remain non-errors and keep their old behavior.
		if errors:
			raise RuntimeError(
				f"DeepEval judge failed on {len(errors)}/{len(scores)} test cases for "
				f"{metric_class.__name__}; partial metric results discarded. First error: "
				f"{errors[0][:500]}"
			)
		n_skipped = sum(1 for s in scores if s is None)
		scored = [float(s) for s in scores if s is not None]
		mean_text = f"{sum(scored) / len(scored):.4f}" if scored else "nan"
		logger.info(
			f"[judge] {metric_class.__name__}: complete "
			f"scored={len(scored)}/{len(scores)} skipped={n_skipped} mean={mean_text}"
		)
		return [s if s is not None else float("nan") for s in scores]
	finally:
		close = getattr(adapter, "close", None)
		if callable(close):
			close()


# ---------------------------------------------------------------------------
# Metric entry points. Each accepts the YAML-declared ``model`` plus arbitrary
# kwargs that are forwarded to ``get_ai_client``. Common kwargs:
#
#   For vLLM judge:          JUDGE_BASE_URL/JUDGE_MODEL env vars, or base_url
#                            plus model: vllm/<served-model-name> in YAML;
#                            max_concurrency defaults to JUDGE_MAX_CONCURRENCY
#                            or 64.
#   For OpenAIClient:         base_url, api_key, max_concurrency, max_retries
#   For ClaudeCodeClient:     effort, thinking, cwd, max_concurrency
#   For CodexCliClient:       effort, sandbox, full_auto, skip_git_repo_check,
#                             ephemeral, extra_config, timeout_s, max_concurrency
# ---------------------------------------------------------------------------

@autorag_metric_loop(fields_to_check=["query", "generation_gt", "retrieved_contents"])
def deepeval_context_precision(
	metric_inputs: List[MetricInput],
	model: str = "openai/gpt-4o",
	threshold: float = 0.5,
	**ai_client_kwargs: Any,
) -> List[float]:
	"""Whether relevant retrieved contexts are ranked higher."""
	return _measure_batch(
		ContextualPrecisionMetric,
		metric_inputs,
		lambda mi: _build_test_case(
			input=mi.query,
			expected_output="\n".join(mi.generation_gt),
			retrieval_context=mi.retrieved_contents,
		),
		model, threshold, ai_client_kwargs,
	)


@autorag_metric_loop(fields_to_check=["query", "generation_gt", "retrieved_contents"])
def deepeval_context_recall(
	metric_inputs: List[MetricInput],
	model: str = "openai/gpt-4o",
	threshold: float = 0.5,
	**ai_client_kwargs: Any,
) -> List[float]:
	"""Extent to which retrieval_context aligns with expected_output."""
	return _measure_batch(
		ContextualRecallMetric,
		metric_inputs,
		lambda mi: _build_test_case(
			input=mi.query,
			expected_output="\n".join(mi.generation_gt),
			retrieval_context=mi.retrieved_contents,
		),
		model, threshold, ai_client_kwargs,
	)


@autorag_metric_loop(fields_to_check=["query", "generated_texts", "retrieved_contents"])
def deepeval_contextual_relevancy(
	metric_inputs: List[MetricInput],
	model: str = "openai/gpt-4o",
	threshold: float = 0.5,
	**ai_client_kwargs: Any,
) -> List[float]:
	"""Overall relevance of retrieval_context to the input query."""
	return _measure_batch(
		ContextualRelevancyMetric,
		metric_inputs,
		lambda mi: _build_test_case(
			input=mi.query,
			actual_output=mi.generated_texts,
			retrieval_context=mi.retrieved_contents,
		),
		model, threshold, ai_client_kwargs,
	)


@autorag_metric_loop(fields_to_check=["query", "generated_texts"])
def deepeval_answer_relevancy(
	metric_inputs: List[MetricInput],
	model: str = "openai/gpt-4o",
	threshold: float = 0.5,
	**ai_client_kwargs: Any,
) -> List[float]:
	"""How relevant the actual_output is to the input query."""
	return _measure_batch(
		AnswerRelevancyMetric,
		metric_inputs,
		lambda mi: _build_test_case(
			input=mi.query,
			actual_output=mi.generated_texts,
		),
		model, threshold, ai_client_kwargs,
	)


@autorag_metric_loop(fields_to_check=["query", "generated_texts", "retrieved_contents"])
def deepeval_faithfulness(
	metric_inputs: List[MetricInput],
	model: str = "openai/gpt-4o",
	threshold: float = 0.5,
	**ai_client_kwargs: Any,
) -> List[float]:
	"""Whether actual_output factually aligns with retrieval_context."""
	return _measure_batch(
		FaithfulnessMetric,
		metric_inputs,
		lambda mi: _build_test_case(
			input=mi.query,
			actual_output=mi.generated_texts,
			retrieval_context=mi.retrieved_contents,
		),
		model, threshold, ai_client_kwargs,
	)


@autorag_metric_loop(fields_to_check=["query", "generated_texts", "generation_gt"])
def deepeval_answer_correctness(
	metric_inputs: List[MetricInput],
	model: str = "openai/gpt-4o",
	threshold: float = 0.5,
	**ai_client_kwargs: Any,
) -> List[float]:
	"""Factual correctness of actual_output vs expected_output (gold answer).

	Uses GEval LLM-judge to compare the generated answer against
	``generation_gt`` from the qa.parquet. Score ∈ [0, 1]: 1 = factually
	matches gold (no hallucination, no missing keypoints), 0 = wholly
	incorrect / contradicts gold.

	This is THE business-target metric — distinct from ``faithfulness``
	(which only checks consistency with the retrieved context, not whether
	the answer is correct vs. ground truth) and from ``answer_relevancy``
	(which only checks topical relatedness to the query, ignoring whether
	the answer is right).
	"""
	return _measure_batch(
		_AnswerCorrectnessMetric,
		metric_inputs,
		lambda mi: _build_test_case(
			input=mi.query,
			actual_output=mi.generated_texts,
			expected_output="\n".join(mi.generation_gt),
		),
		model, threshold, ai_client_kwargs,
	)
