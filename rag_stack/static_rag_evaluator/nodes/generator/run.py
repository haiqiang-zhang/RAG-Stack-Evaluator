# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
import os
from typing import List, Dict, Union

import pandas as pd

logger = logging.getLogger("RAG-Stack")

from rag_stack.static_rag_evaluator.evaluation import evaluate_generation
from rag_stack.static_rag_evaluator.evaluation.util import cast_metrics
from rag_stack.static_rag_evaluator.schema.metricinput import MetricInput
from rag_stack.static_rag_evaluator.strategy import measure_speed
from rag_stack.static_rag_evaluator.utils.util import to_list
from rag_stack.security import safe_dataframe_to_csv


def run_generator_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the single generator module supplied by the upstream sampler.

	Greedy multi-module selection has been removed. Exactly one module is expected.
	"""
	if len(modules) != 1:
		raise ValueError(
			f"generator expects exactly one module after sampling, got {len(modules)}"
		)
	module = modules[0]
	module_param = module_params[0]

	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	project_dir = os.environ["PROJECT_DIR"]
	node_dir = os.path.join(node_line_dir, "generator")
	if not os.path.exists(node_dir):
		os.makedirs(node_dir)
	qa_data = pd.read_parquet(
		os.path.join(project_dir, "data", "qa.parquet"), engine="pyarrow"
	)
	if "generation_gt" not in qa_data.columns:
		raise ValueError("You must have 'generation_gt' column in qa.parquet.")

	# ── Generation-input fingerprint dedup (M3, 07-07; env-gated OFF) ────
	# Identical (queries, final contexts, prompts, generator semantics) ⇒ the
	# answer/judge distributions are the SAME random variable as the donor
	# eval's — inherit its answers (and, via the run-root marker, its quality)
	# instead of re-sampling them at full GT price. Content-keyed, so sampled
	# upstream stages (QE at temperature>0) self-invalidate via changed
	# contexts. See rag_stack/static_rag_evaluator/gen_fingerprint.py.
	from rag_stack.static_rag_evaluator import gen_fingerprint as _fp
	fp = None
	run_root = os.path.dirname(node_line_dir)
	if _fp.enabled() and "prompts" in previous_result.columns:
		try:
			fp = _fp.fingerprint(
				queries=previous_result["query"].astype(str).tolist()
				if "query" in previous_result.columns else [],
				contexts=previous_result["retrieved_contents"].tolist()
				if "retrieved_contents" in previous_result.columns else [],
				prompts=previous_result["prompts"].astype(str).tolist(),
				module_name=getattr(module, "__name__", str(module)),
				module_param=module_param,
			)
		except Exception as exc:  # noqa: BLE001 — dedup must never break an eval
			logger.warning(f"[gen-fp] fingerprint failed ({exc}); running normally")
			fp = None
	donor_rec = _fp.load_complete_record(project_dir, fp) if fp else None

	if donor_rec is not None and len(donor_rec["answers"]) == len(previous_result):
		logger.info(
			f"[gen-fp] HIT {fp[:12]} (donor={donor_rec.get('donor', '?')}) — "
			"inheriting answers; generation skipped"
		)
		result = pd.DataFrame({
			"generated_texts": list(donor_rec["answers"]),
			# token-id lists reconstructed at the DONOR's exact lengths so the
			# trace's output token counts (len of ids) reproduce the donor's.
			"generated_tokens": [[0] * int(n) for n in donor_rec["token_counts"]],
			"generated_log_probs": [None] * len(donor_rec["answers"]),
		})
		execution_time = 1e-9
		_fp.write_marker(run_root, "fp_hit.json",
						 {"fp": fp, "donor": donor_rec.get("donor", "?")})
	else:
		result, execution_time = measure_speed(
			module.run_evaluator,
			project_dir=project_dir,
			previous_result=previous_result,
			**module_param,
		)
		if fp is not None and "generated_texts" in result.columns:
			saved = _fp.save_answers(
				project_dir, fp,
				answers=result["generated_texts"].tolist(),
				token_counts=[len(t) for t in result["generated_tokens"].tolist()],
				donor=os.path.basename(run_root.rstrip(os.sep)) or run_root,
			)
			if saved:
				_fp.write_marker(run_root, "fp_pending.json", {"fp": fp})
	average_time = execution_time / len(result)
	token_usage = result["generated_tokens"].apply(len).mean()

	generation_gt = to_list(qa_data["generation_gt"].tolist())
	queries = to_list(qa_data["query"].tolist())
	metric_inputs = [
		MetricInput(generation_gt=gen_gt, query=query)
		for gen_gt, query in zip(generation_gt, queries)
	]

	metric_names = []
	if strategies.get("metrics"):
		metric_names, _ = cast_metrics(strategies.get("metrics"))
		result = evaluate_generator_node(
			result, metric_inputs, strategies.get("metrics")
		)

	filepath = os.path.join(node_dir, "0.parquet")
	result.to_parquet(filepath, index=False)
	filename = os.path.basename(filepath)

	summary_df = pd.DataFrame(
		{
			"filename": [filename],
			"module_name": [module.__name__],
			"module_params": [module_param],
			"execution_time": [average_time],
			"average_output_token": [token_usage],
			**{metric: [result[metric].mean()] for metric in metric_names},
		}
	)

	best_result = pd.concat([previous_result, result], axis=1)

	# Orthogonal trace recording: emit one "generate" call per query (output tokens
	# EXACT from the vLLM token ids; input tokens from the prompt). Mode-agnostic —
	# fires identically in measured and quality-only. No-op when no recorder bound.
	_record_generate(best_result, result, module_param)

	safe_dataframe_to_csv(summary_df, os.path.join(node_dir, "summary.csv"), index=False)
	best_result.to_parquet(
		os.path.join(node_dir, f"best_{os.path.splitext(filename)[0]}.parquet"),
		index=False,
	)
	return best_result


def _record_generate(best_result: pd.DataFrame, result: pd.DataFrame, module_param: Dict) -> None:
	"""Emit one 'generate' trace call per query (no-op when no recorder bound). Output
	tokens are EXACT (``len(generated_tokens)`` = vLLM token ids); input tokens are formally
	tokenized from the prompt; bytes are exact UTF-8. Keyed by the permanent ``__qid__``."""
	from rag_stack.static_rag_evaluator import recording as _rec
	if "__qid__" not in best_result.columns or "generated_tokens" not in result.columns:
		return
	qids = best_result["__qid__"].tolist()
	model_id = module_param.get("model") or module_param.get("llm")
	if "prompts" in best_result.columns:
		prompts = best_result["prompts"].tolist()
	elif "query" in best_result.columns:
		prompts = best_result["query"].astype(str).tolist()
	else:
		prompts = ["" for _ in qids]
	out_texts = (best_result["generated_texts"].tolist()
				 if "generated_texts" in best_result.columns else None)
	_rec.record_io("generator", qids, prompts, out_texts=out_texts,
				   out_token_ids=result["generated_tokens"].tolist(), model_id=model_id)


def evaluate_generator_node(
	result_df: pd.DataFrame,
	metric_inputs: List[MetricInput],
	metrics: Union[List[str], List[Dict]],
):
	@evaluate_generation(metric_inputs=metric_inputs, metrics=metrics)
	def evaluate_generation_module(df: pd.DataFrame):
		return (
			df["generated_texts"].tolist(),
			df["generated_tokens"].tolist(),
			df["generated_log_probs"].tolist(),
		)

	return evaluate_generation_module(result_df)
