# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from itertools import chain
from typing import List, Tuple

import pandas as pd

from rag_stack.cost_model.reranker_policy import (
	MONOT5_FORWARD_EXECUTION_SCHEMA,
)
from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.base import BasePassageReranker
from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.dp import (
	rerank_flatten_apply_dp,
)
from rag_stack_evaluator.static_rag_evaluator.utils.util import (
	make_batch,
	sort_by_scores,
	select_top_k,
	result_to_dataframe,
	pop_params,
)

prediction_tokens = {
	"castorini/monot5-base-msmarco": ["▁false", "▁true"],
	"castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
	"castorini/monot5-large-msmarco": ["▁false", "▁true"],
	"castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
	"castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
	"castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
	"castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
	"unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
	"unicamp-dl/ptt5-base-pt-msmarco-10k-v2": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-pt-msmarco-100k-v2": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2": ["▁não", "▁sim"],
	"unicamp-dl/mt5-base-en-pt-msmarco-v2": ["▁no", "▁yes"],
	"unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
	"unicamp-dl/mt5-base-en-pt-msmarco-v1": ["▁no", "▁yes"],
	"unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
	"unicamp-dl/ptt5-base-pt-msmarco-10k-v1": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-pt-msmarco-100k-v1": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1": ["▁não", "▁sim"],
	"unicamp-dl/mt5-3B-mmarco-en-pt": ["▁", "▁true"],
	"unicamp-dl/mt5-13b-mmarco-100k": ["▁", "▁true"],
}

class MonoT5(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		model_name: str = "castorini/monot5-3b-msmarco-10k",
		*args,
		**kwargs,
	):
		"""
		Initialize the MonoT5 reranker.

		:param project_dir: The project directory
		:param model_name: The name of the MonoT5 model to use for reranking
			Note: default model name is 'castorini/monot5-3b-msmarco-10k'
				If there is a '/' in the model name parameter,
				when we create the file to store the results, the path will be twisted because of the '/'.
				Therefore, it will be received as '_' instead of '/'.
		:param kwargs: The extra arguments for the MonoT5 reranker
		"""
		super().__init__(project_dir)
		try:
			import torch
			from transformers import T5Tokenizer, T5ForConditionalGeneration
		except ImportError:
			raise ImportError("For using MonoT5 Reranker, please install torch first.")
		# replace '_' to '/'
		if "_" in model_name:
			model_name = model_name.replace("_", "/")
		# Cache-aware load: on the perf path (a ModelCache is current) the
		# (tokenizer, model) bundle is loaded ONCE and reused across trials,
		# placed on the injected per-component device. On the quality path
		# (cache is None) the legacy fresh-per-call load is preserved.
		from rag_stack_evaluator.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		kwargs.pop("cache", None)  # strip stale entries if any
		device_override = kwargs.pop("device", None)
		devices = kwargs.pop("devices", None)
		default_device = device_override or (
			"cuda" if torch.cuda.is_available() else "cpu"
		)
		model_params = pop_params(T5ForConditionalGeneration.from_pretrained, kwargs)

		def _build(mt, mn, dev):
			# Fast (Rust) tokenizer: the slow SentencePiece class tokenized
			# every query-document pair in Python and was a visible slice of
			# the per-forward wall (07-10 overhead audit). Same vocab, same
			# piece ids — convert_tokens_to_ids('▁false'/'▁true') is
			# unchanged. Slow class kept as fallback for environments
			# without the tokenizers wheel.
			try:
				from transformers import T5TokenizerFast
				tok = T5TokenizerFast.from_pretrained(mn)
			except Exception:  # noqa: BLE001 — fall back to the slow class
				tok = T5Tokenizer.from_pretrained(mn)
			mdl = (
				T5ForConditionalGeneration.from_pretrained(mn, **model_params)
				.eval()
				.to(dev)
			)
			return (tok, mdl)

		self._load_replicas(
			cache,
			component="monot5",
			model_name=model_name,
			device=default_device,
			devices=devices,
			factory=_build,
		)
		self.tokenizer, self.model = self._replicas[0]

		token_false, token_true = prediction_tokens[model_name]
		self.token_false_id = self.tokenizer.convert_tokens_to_ids(token_false)
		self.token_true_id = self.tokenizer.convert_tokens_to_ids(token_true)
		self._last_forward_execution_report = None

	def __del__(self):
		if getattr(self, "_cache_owned", False):
			# Cache owns the model lifetime; don't free here.
			try:
				super().__del__()
			except Exception:
				pass
			return
		if hasattr(self, "model"):
			del self.model
		if hasattr(self, "tokenizer"):
			del self.tokenizer
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.get("top_k", 3)
		batch = kwargs.get("batch", 64)
		return self._pure(queries, contents, ids, top_k, batch)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents based on their relevance to a query using MonoT5.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved

		:param batch: The number of queries to be processed in a batch
		:return: tuple of lists containing the reranked contents, ids, and scores
		"""
		# Retrieve the tokens used by the model to represent false and true predictions

		nested_list = [
			list(map(lambda x: [f"Query: {query} Document: {x}"], content_list))
			for query, content_list in zip(queries, contents_list)
		]

		execution_reports = []
		self._last_forward_execution_report = None
		replicas_with_devices = list(zip(self._replicas, self._replica_devices))

		def _run_replica(input_texts, model, batch_size):
			(tokenizer, replica_model), replica_device = model
			report = {}
			try:
				return monot5_run_model(
					input_texts,
					model=replica_model,
					batch_size=batch_size,
					tokenizer=tokenizer,
					device=replica_device,
					token_false_id=self.token_false_id,
					token_true_id=self.token_true_id,
					execution_report=report,
				)
			finally:
				if report:
					execution_reports.append(report)

		try:
			rerank_scores = rerank_flatten_apply_dp(
				_run_replica,
				nested_list,
				replicas_with_devices,
				batch_size=batch,
			)
		finally:
			if execution_reports:
				self._last_forward_execution_report = {
					"schema": MONOT5_FORWARD_EXECUTION_SCHEMA,
					"requested_forward_microbatch": max(
						int(report["requested_forward_microbatch"])
						for report in execution_reports
					),
					"successful_forward_microbatches": [
						size
						for report in execution_reports
						for size in report["successful_forward_microbatches"]
					],
					"actual_forward_microbatch": max(
						int(report["actual_forward_microbatch"])
						for report in execution_reports
					),
					"oom_fallback_count": sum(
						int(report["oom_fallback_count"])
						for report in execution_reports
					),
					"failed_forward_microbatches": [
						size
						for report in execution_reports
						for size in report["failed_forward_microbatches"]
					],
				}

		df = pd.DataFrame(
			{
				"contents": contents_list,
				"ids": ids_list,
				"scores": rerank_scores,
			}
		)
		df[["contents", "ids", "scores"]] = df.apply(
			sort_by_scores, axis=1, result_type="expand"
		)
		results = select_top_k(df, ["contents", "ids", "scores"], top_k)

		return (
			results["contents"].tolist(),
			results["ids"].tolist(),
			results["scores"].tolist(),
		)

	def pop_last_forward_execution_report(self):
		"""Return and clear telemetry for the most recent production call."""

		report = self._last_forward_execution_report
		self._last_forward_execution_report = None
		if report is None:
			return None
		return {
			key: list(value) if isinstance(value, list) else value
			for key, value in report.items()
		}


def monot5_run_model(
	input_texts,
	model,
	batch_size: int,
	tokenizer,
	device,
	token_false_id,
	token_true_id,
	execution_report=None,
):
	try:
		import torch
	except ImportError:
		raise ImportError("For using MonoT5 Reranker, please install torch first.")
	def _score_batch(texts: list) -> list:
		input_encodings = tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=512,
			return_tensors="pt",
		).to(device)
		with torch.no_grad():
			outputs = model.generate(
				input_ids=input_encodings["input_ids"],
				attention_mask=input_encodings["attention_mask"],
				output_scores=True,
				return_dict_in_generate=True,
			)
		# Logits for the 'false'/'true' tokens → P(true) per pair.
		logits = outputs.scores[-1][:, [token_false_id, token_true_id]]
		probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
		out = probs.tolist()
		# Drop forward refs NOW: a 3B fp32 forward at batch 64 peaks ~20 GiB
		# alloc (~30+ GiB reserved with varying shapes) — keeping the last
		# batch's graph alive doubles the co-residency with the vLLM engine.
		del outputs, logits, probs, input_encodings
		return out

	requested_forward_microbatch = max(1, int(batch_size))
	report = {
		"schema": MONOT5_FORWARD_EXECUTION_SCHEMA,
		"requested_forward_microbatch": requested_forward_microbatch,
		"successful_forward_microbatches": [],
		"actual_forward_microbatch": 0,
		"oom_fallback_count": 0,
		"failed_forward_microbatches": [],
	}

	def _publish_report() -> None:
		if execution_report is None:
			return
		execution_report.clear()
		execution_report.update({
			key: list(value) if isinstance(value, list) else value
			for key, value in report.items()
		})

	_publish_report()
	batch_input_texts = make_batch(input_texts, requested_forward_microbatch)
	results = []
	for batch_texts in batch_input_texts:
		flattened_batch_texts = list(chain.from_iterable(batch_texts))
		# Adaptive OOM fallback: halve the forward batch until it fits (scores
		# are batching-invariant, so this never changes results — it only
		# bounds activation memory when the GPU is shared with a live vLLM
		# engine). Raises only if even batch=1 cannot fit.
		size = max(1, len(flattened_batch_texts))
		while True:
			attempt_sizes: list[int] = []
			attempted_forward_size = size
			try:
				batch_scores: list = []
				for i in range(0, len(flattened_batch_texts), size):
					forward_texts = flattened_batch_texts[i:i + size]
					attempted_forward_size = len(forward_texts)
					batch_scores.extend(_score_batch(forward_texts))
					attempt_sizes.append(len(forward_texts))
				results.extend(batch_scores)
				report["successful_forward_microbatches"].extend(attempt_sizes)
				report["actual_forward_microbatch"] = max(
					report["successful_forward_microbatches"]
				)
				_publish_report()
				break
			except torch.cuda.OutOfMemoryError:
				report["oom_fallback_count"] += 1
				report["failed_forward_microbatches"].append(
					attempted_forward_size
				)
				_publish_report()
				torch.cuda.empty_cache()
				if size == 1:
					raise
				size = max(1, size // 2)
	_publish_report()
	return results
