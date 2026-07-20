# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from typing import List, Optional

import pandas as pd

from rag_stack.static_rag_evaluator.nodes.passagecompressor.base import BasePassageCompressor
from rag_stack.static_rag_evaluator.utils.util import pop_params, result_to_dataframe

# TODO: Parallel Processing Refactoring at #460


def _patch_llmlingua_for_new_transformers() -> None:
	"""Patch llmlingua.PromptCompressor.get_ppl for transformers >= 4.43.

	llmlingua was written against the legacy ``past_key_values`` API where
	it was a ``Tuple[Tuple[Tensor, Tensor], ...]``. Modern transformers
	(>= 4.43) require ``past_key_values`` to be a ``Cache`` object exposing
	``.get_seq_length()``; passing a list/tuple raises::

	    AttributeError: 'list' object has no attribute 'get_seq_length'

	We don't want to rewrite llmlingua. Instead, swap a thin wrapper that
	converts legacy<->Cache at the ``self.model(...)`` boundary so the rest
	of ``iterative_compress_prompt`` (which slices the kv-cache as a list
	of (k, v) tensor tuples) keeps working unchanged.
	"""
	import torch
	from llmlingua import PromptCompressor
	from transformers import DynamicCache

	if getattr(PromptCompressor.get_ppl, "_patched_for_cache", False):
		return  # idempotent

	def get_ppl(
		self,
		text: str,
		granularity: str = "sentence",
		input_ids=None,
		attention_mask=None,
		past_key_values=None,
		return_kv=False,
		end=None,
		condition_mode: str = "none",
		condition_pos_id: int = 0,
	):
		if input_ids is None:
			tokenized_text = self.tokenizer(text, return_tensors="pt")
			input_ids = tokenized_text["input_ids"].to(self.device)
			attention_mask = tokenized_text["attention_mask"].to(self.device)
		# `past_key_values` arrives in legacy tuple-of-tuples shape (None on
		# the first call). Compute past_length from the legacy form before
		# we wrap.
		if past_key_values is not None and len(past_key_values) > 0:
			past_length = past_key_values[0][0].shape[2]
		else:
			past_length = 0
		if end is None:
			end = input_ids.shape[1]
		end = min(end, past_length + self.max_position_embeddings)
		# Convert legacy -> Cache for the model call.
		cache_for_model = None
		if past_key_values is not None and len(past_key_values) > 0:
			cache_for_model = DynamicCache.from_legacy_cache(past_key_values)
		with torch.no_grad():
			response = self.model(
				input_ids[:, past_length:end],
				attention_mask=attention_mask[:, :end],
				past_key_values=cache_for_model,
				use_cache=True,
			)
			# Cache -> legacy tuple so the surrounding iterative-compress
			# code (which expects `for k, v in past_key_values: ...`)
			# keeps working.
			new_kv = response.past_key_values
			if hasattr(new_kv, "to_legacy_cache"):
				past_key_values = new_kv.to_legacy_cache()
			else:
				past_key_values = new_kv
		shift_logits = response.logits[..., :-1, :].contiguous()
		shift_labels = input_ids[..., past_length + 1 : end].contiguous()
		active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
		active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
		active_labels = shift_labels.view(-1)[active]
		loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
		loss = loss_fct(active_logits, active_labels)
		if condition_mode == "before":
			loss = loss[:condition_pos_id]
		elif condition_mode == "after":
			loss = loss[condition_pos_id:]
		res = loss.mean() if granularity == "sentence" else loss
		return (res, past_key_values) if return_kv else res

	get_ppl._patched_for_cache = True
	PromptCompressor.get_ppl = get_ppl


_patch_llmlingua_for_new_transformers()


def _align_sharded_device(prompt_compressor, fallback: str) -> None:
	"""When the ~7B model is sharded across GPUs (``device_map="auto"``),
	llmlingua leaves ``PromptCompressor.device`` at its constructor default
	(``"cuda"`` == cuda:0), but Accelerate may place the input-embedding shard on
	a non-zero ordinal. The patched ``get_ppl`` moves ``input_ids`` to
	``self.device``, so a mismatch → ``RuntimeError: tensors on different
	devices`` on the first forward. Re-point ``device`` at the shard that holds
	the embedding (its ``hf_device_map`` lead), falling back to ``fallback``."""
	model = getattr(prompt_compressor, "model", None)
	dmap = getattr(model, "hf_device_map", None) if model is not None else None
	lead = None
	if isinstance(dmap, dict) and dmap:
		# Prefer the input-embedding's device (where get_ppl must send inputs).
		for layer, dev in dmap.items():
			if "embed" in layer:
				lead = dev
				break
		if lead is None:
			lead = next(iter(dmap.values()))
		lead = f"cuda:{lead}" if isinstance(lead, int) else str(lead)
	try:
		prompt_compressor.device = lead or fallback
	except Exception:  # noqa: BLE001 — best-effort; never block the trial
		pass


class LongLLMLingua(BasePassageCompressor):
	def __init__(
		self, project_dir: str, model_name: str = "NousResearch/Llama-2-7b-hf", **kwargs
	):
		try:
			from llmlingua import PromptCompressor
		except ImportError:
			raise ImportError(
				"LongLLMLingua is not installed. Please install it by running `pip install llmlingua`."
			)

		super().__init__(project_dir)
		# Cache-aware load: PromptCompressor wraps a full ~7B LLM. On the perf
		# path (a ModelCache is current) load it ONCE + reuse across trials,
		# placed on the injected device via PromptCompressor's `device_map`. On
		# the quality path (cache is None) the legacy fresh load is preserved.
		from rag_stack.static_rag_evaluator.measured.cache import get_current
		cache = get_current()
		kwargs.pop("cache", None)
		device_override = kwargs.pop("device", None)
		devices = kwargs.pop("devices", None)  # measured multi-GPU device list
		model_init_params = pop_params(PromptCompressor.__init__, kwargs)
		devs = [str(d) for d in (devices or ([device_override] if device_override else [])) if d]
		if len(devs) > 1:
			# Multi-GPU: shard the ~7B model across the engine's devices via HF
			# Accelerate (device_map="auto" bounded to those GPU ordinals) — the
			# measured analog of the cost model's compressor TP.
			ordinals = [int(d.split(":")[-1]) for d in devs if ":" in d]
			model_init_params.setdefault("device_map", "auto")
			mcfg = dict(model_init_params.get("model_config") or {})
			mcfg.setdefault("max_memory", {o: "20GiB" for o in ordinals})
			model_init_params["model_config"] = mcfg
		elif device_override is not None:
			model_init_params.setdefault("device_map", device_override)

		def _build(mt, mn, dev):
			return PromptCompressor(model_name=mn, **model_init_params)

		if cache is not None:
			self._cache_owned = True
			# Reuse the generic reranker object cache (keyed by
			# (component, model_name, device)) — it holds any cached model.
			self.llm_lingua = cache.get_reranker(
				component="longllmlingua",
				model_name=model_name,
				device=device_override or (devs[0] if devs else "cuda:0"),
				factory=_build,
			)
		else:
			self._cache_owned = False
			self.llm_lingua = PromptCompressor(
				model_name=model_name, **model_init_params
			)
		# Sharded multi-GPU load: re-point the compressor's device at the
		# embedding shard so get_ppl's input placement matches the model.
		if len(devs) > 1:
			_align_sharded_device(self.llm_lingua, devs[0])

	def __del__(self):
		if getattr(self, "_cache_owned", False):
			try:
				super().__del__()
			except Exception:
				pass
			return
		if hasattr(self, "llm_lingua"):
			del self.llm_lingua
		super().__del__()

	@result_to_dataframe(["retrieved_contents"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, retrieved_contents = self.cast_to_run(previous_result)
		results = self._pure(queries, retrieved_contents, **kwargs)
		return list(map(lambda x: [x], results))

	def _pure(
		self,
		queries: List[str],
		contents: List[List[str]],
		instructions: Optional[str] = None,
		target_token: int = 300,
		**kwargs,
	) -> List[str]:
		"""
		Compresses the retrieved texts using LongLLMLingua.
		For more information, visit https://github.com/microsoft/LLMLingua.

		:param queries: The queries for retrieved passages.
		:param contents: The contents of retrieved passages.
		:param model_name: The model name to use for compression.
		    The default is "NousResearch/Llama-2-7b-hf".
		:param instructions: The instructions for compression.
		    Default is None. When it is None, it will use default instructions.
		:param target_token: The target token for compression.
		    Default is 300.
		:param kwargs: Additional keyword arguments.
		:return: The list of compressed texts.
		"""
		if instructions is None:
			instructions = "Given the context, please answer the final question"
		results = [
			llmlingua_pure(
				query, contents_, self.llm_lingua, instructions, target_token, **kwargs
			)
			for query, contents_ in zip(queries, contents)
		]

		return results


def llmlingua_pure(
	query: str,
	contents: List[str],
	llm_lingua,
	instructions: str,
	target_token: int = 300,
	**kwargs,
) -> str:
	"""
	Return the compressed text.

	:param query: The query for retrieved passages.
	:param contents: The contents of retrieved passages.
	:param llm_lingua: The llm instance, that will be used to compress.
	:param instructions: The instructions for compression.
	:param target_token: The target token for compression.
	    Default is 300.
	:param kwargs: Additional keyword arguments.
	:return: The compressed text.
	"""
	try:
		from llmlingua import PromptCompressor
	except ImportError:
		raise ImportError(
			"LongLLMLingua is not installed. Please install it by running `pip install llmlingua`."
		)
	# split by "\n\n" (recommended by LongLLMLingua authors)
	new_context_texts = [c for context in contents for c in context.split("\n\n")]
	compress_prompt_params = pop_params(PromptCompressor.compress_prompt, kwargs)
	compressed_prompt = llm_lingua.compress_prompt(
		new_context_texts,
		question=query,
		instruction=instructions,
		rank_method="longllmlingua",
		target_token=target_token,
		**compress_prompt_params,
	)
	compressed_prompt_txt = compressed_prompt["compressed_prompt"]

	# separate out the question and instruction
	result = "\n\n".join(compressed_prompt_txt.split("\n\n")[1:-1])

	return result
