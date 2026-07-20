"""Per-query performance instrumentation aligned with the cost-model schema.

Each `QueryPerf` records timestamps for one query through the e2e pipeline.
`summarize()` aggregates a batch of `QueryPerf` records into a dict whose
keys mirror the columns of `rag_stack/cost_model`'s `rago_sweep.csv`
(latency_s, latency_s_ttft, latency_s_tpot, qps, qps_per_chip, etc.).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class QueryPerf:
	"""Timestamps and counts for a single query.

	`first_token_ts` may be None for non-streaming pipelines (TTFT/TPOT then
	reduce to e2e latency).
	"""

	request_send_ts: float
	last_token_ts: float
	n_output_tokens: int
	first_token_ts: Optional[float] = None
	per_stage_times: Dict[str, float] = field(default_factory=dict)
	agent_generate_calls: int = 0
	agent_retrieval_calls: int = 0
	agent_truncated: bool = False

	@property
	def e2e_s(self) -> float:
		return self.last_token_ts - self.request_send_ts

	@property
	def ttft_s(self) -> float:
		if self.first_token_ts is None:
			return self.e2e_s
		return self.first_token_ts - self.request_send_ts

	@property
	def tpot_s(self) -> float:
		if self.first_token_ts is None or self.n_output_tokens <= 1:
			return 0.0
		return (self.last_token_ts - self.first_token_ts) / (self.n_output_tokens - 1)


def _percentile_summary(values: List[float]) -> Dict[str, float]:
	if not values:
		return {"median": 0.0, "mean": 0.0, "p95": 0.0}
	arr = np.asarray(values, dtype=float)
	return {
		"median": float(np.median(arr)),
		"mean": float(arr.mean()),
		"p95": float(np.percentile(arr, 95)),
	}


def summarize(
	perfs: List[QueryPerf],
	total_wall_clock_s: float,
	n_chips: int = 1,
	throughput_queries: Optional[int] = None,
	throughput_output_tokens: Optional[int] = None,
) -> Dict[str, object]:
	"""Aggregate per-query records into the cost-model column schema.

	Output keys match `rag_stack/cost_model` `rago_sweep.csv` columns where
	applicable so the baseline's `eval_history.csv` can be compared directly
	against cost-model predictions axis-by-axis.
	"""
	if not perfs:
		return {
			"latency_s": _percentile_summary([]),
			"latency_s_ttft": _percentile_summary([]),
			"latency_s_tpot": _percentile_summary([]),
			"qps": 0.0,
			"qps_per_chip": 0.0,
			"output_tokens_per_s": 0.0,
			"per_stage_latency_s": {},
		}

	e2e = [p.e2e_s for p in perfs]
	ttft = [p.ttft_s for p in perfs]
	tpot = [p.tpot_s for p in perfs]
	total_output_tokens = sum(p.n_output_tokens for p in perfs)
	n_queries = len(perfs)
	rate_queries = n_queries if throughput_queries is None else throughput_queries
	rate_output_tokens = (
		total_output_tokens
		if throughput_output_tokens is None else throughput_output_tokens
	)
	qps = rate_queries / total_wall_clock_s if total_wall_clock_s > 0 else 0.0
	output_tokens_per_s = (
		rate_output_tokens / total_wall_clock_s if total_wall_clock_s > 0 else 0.0
	)

	# Per-stage: median of recorded times across queries (only stages present)
	stage_keys: set = set()
	for p in perfs:
		stage_keys.update(p.per_stage_times)
	per_stage_latency_s = {
		stage: float(np.median([p.per_stage_times.get(stage, 0.0) for p in perfs]))
		for stage in sorted(stage_keys)
	}
	agent_generate_calls = [p.agent_generate_calls for p in perfs]
	agent_retrieval_calls = [p.agent_retrieval_calls for p in perfs]
	agent_truncated_count = sum(1 for p in perfs if p.agent_truncated)

	out = {
		"latency_s": _percentile_summary(e2e),
		"latency_s_ttft": _percentile_summary(ttft),
		"latency_s_tpot": _percentile_summary(tpot),
		"qps": float(qps),
		"qps_per_chip": float(qps / n_chips) if n_chips > 0 else 0.0,
		"output_tokens_per_s": float(output_tokens_per_s),
		"per_stage_latency_s": per_stage_latency_s,
	}
	if any(agent_generate_calls) or any(agent_retrieval_calls) or agent_truncated_count:
		out["agent_generate_calls"] = _percentile_summary(
			[float(v) for v in agent_generate_calls]
		)
		out["agent_retrieval_calls"] = _percentile_summary(
			[float(v) for v in agent_retrieval_calls]
		)
		out["agent_truncated_count"] = int(agent_truncated_count)
		out["agent_truncated_fraction"] = float(agent_truncated_count / n_queries)
	return out
