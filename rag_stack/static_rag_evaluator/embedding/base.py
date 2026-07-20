# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import logging
import os
import sys

from random import random
from typing import List, Optional, Union, Dict

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from rag_stack.static_rag_evaluator import LazyInit
from rag_stack.static_rag_evaluator.embedding.vllm import VllmEmbedding

logger = logging.getLogger("RAG-Stack")


class MockEmbeddingRandom(MockEmbedding):
	"""Mock embedding with random vectors."""

	def _get_vector(self) -> List[float]:
		return [random() for _ in range(self.embed_dim)]


embedding_models = {
	# llama index
	"openai": LazyInit(
		OpenAIEmbedding
	),  # default model is OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
	"openai_embed_3_large": LazyInit(
		OpenAIEmbedding, model_name=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE
	),
	"openai_embed_3_small": LazyInit(
		OpenAIEmbedding, model_name=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
	),
	"mock": LazyInit(MockEmbeddingRandom, embed_dim=768),
	"ollama": LazyInit(OllamaEmbedding),
	# openai like
	"openai_like": LazyInit(OpenAILikeEmbedding),
	"vllm": LazyInit(VllmEmbedding),
}

# Captured below when the local (GPU) embedding deps are installed. Used by
# set_embedding_device() to detect which registry factories accept a `device`
# kwarg (only the HuggingFace ones do).
_HF_EMBEDDING_CLS = None

def _hf_hub_cache_dir() -> str:
	"""The ONE model cache every HF-backed component shares.

	llama_index's HuggingFaceEmbedding otherwise downloads into its OWN
	cache folder (LLAMA_INDEX_CACHE_DIR / ~/.cache/llama_index) — a second
	copy of the same checkpoints that silently diverges from the hub cache
	the rest of the stack uses. An incomplete snapshot in that second
	location took down every measured retrieval worker once the box lost
	network (msmarco smac s45 evals 0030-0032): the hub copy was complete
	the whole time. Point the embeddings at the hub cache explicitly."""
	hub = os.environ.get("HF_HUB_CACHE")
	if hub:
		return hub
	home = os.environ.get("HF_HOME")
	if home:
		return os.path.join(home, "hub")
	return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


try:
	# you can use your own model in this way.
	from llama_index.embeddings.huggingface import HuggingFaceEmbedding

	_HF_EMBEDDING_CLS = HuggingFaceEmbedding

	embedding_models["huggingface_baai_bge_small"] = LazyInit(
		HuggingFaceEmbedding, model_name="BAAI/bge-small-en-v1.5",
		cache_folder=_hf_hub_cache_dir(),
	)
	embedding_models["huggingface_cointegrated_rubert_tiny2"] = LazyInit(
		HuggingFaceEmbedding, model_name="cointegrated/rubert-tiny2",
		cache_folder=_hf_hub_cache_dir(),
	)
	embedding_models["huggingface_all_mpnet_base_v2"] = LazyInit(
		HuggingFaceEmbedding,
		model_name="sentence-transformers/all-mpnet-base-v2",
		max_length=512,
		cache_folder=_hf_hub_cache_dir(),
	)
	embedding_models["huggingface_bge_m3"] = LazyInit(
		HuggingFaceEmbedding, model_name="BAAI/bge-m3",
		cache_folder=_hf_hub_cache_dir(),
	)
	embedding_models["huggingface_multilingual_e5_large"] = LazyInit(
		HuggingFaceEmbedding, model_name="intfloat/multilingual-e5-large-instruct",
		cache_folder=_hf_hub_cache_dir(),
	)
except ImportError:
	logger.info(
		"You are using API version of RAG-Stack."
		"To use local version, run pip install '.[gpu]'"
	)


# Native output dimension per FIXED-dim local embedding model. Single source of
# truth for DERIVING ``embedding_dim`` from a chosen ``embedding_model`` so the
# dim never has to be hand-set in YAML (and can't drift from the model) — used
# by the search-space resolver (resolve_vectordbs) and the config validator.
# Only fixed-dim local (HuggingFace) models live here; API/MRL models
# (openai*, openai_like, mock) have CONFIGURABLE dims via ``embedding_dim`` and
# are intentionally absent → those keep their explicit YAML ``embedding_dim``.
# Keys are the rag-stack registry names (the embedding_models dict above), what
# a vectordb config carries in ``embedding_model:``.
EMBEDDING_MODEL_DIMS = {
	"huggingface_baai_bge_small": 384,           # BAAI/bge-small-en-v1.5 (BERT-small)
	"huggingface_all_mpnet_base_v2": 768,        # sentence-transformers/all-mpnet-base-v2
	"huggingface_bge_m3": 1024,                  # BAAI/bge-m3 (XLM-R-large backbone)
	"huggingface_multilingual_e5_large": 1024,   # intfloat/multilingual-e5-large-instruct
	"huggingface_cointegrated_rubert_tiny2": 312,  # cointegrated/rubert-tiny2 (RU)
}


def set_embedding_device(device: Optional[str]) -> None:
	"""Pin the HuggingFace embedding factories to ``device`` so the NEXT build
	(both query-time and ingest-time) lands the model on that GPU instead of the
	llama_index default (``cuda`` → cuda:0).

	MEASURED-MODE ONLY. The resolved system layout splits the retrieval node into
	GPU ``encode`` and CPU ``retrieval`` stages; this pins the embedding to the
	``encode`` device. Without this the embedding always loads on cuda:0 and
	collides with whatever vLLM engine sits there. The quality / cost-model path
	never calls this, so the registry stays device-less (unchanged default
	behavior).

	Mutates the registry ``LazyInit._kwargs`` and clears any cached instance so a
	device change forces a rebuild on the new card (``_release_gpu_memory`` also
	clears instances per trial). Only HuggingFace factories are touched — OpenAI
	/ Ollama / mock embeddings take no ``device``.
	"""
	if _HF_EMBEDDING_CLS is None:
		return
	for lazy in embedding_models.values():
		if isinstance(lazy, LazyInit) and lazy._factory is _HF_EMBEDDING_CLS:
			if lazy._kwargs.get("device") != device:
				lazy._kwargs = {**lazy._kwargs, "device": device}
				lazy._instance = None


def build_huggingface_embedding_replicas(
	config: Union[str, Dict, List[Dict]],
	devices: List[str],
	*,
	first_replica=None,
	embedding_dim: Optional[int] = None,
) -> List[object]:
	"""Build one independent local-HF embedding model per declared device.

	Measured semantic retrieval constructs its first embedding through the
	existing registry after :func:`set_embedding_device` pins it to
	``devices[0]``.  This helper reuses that object and directly invokes the
	registry factory for the remaining devices, avoiding ``LazyInit``'s global
	single-instance cache.  Non-HuggingFace embeddings fail explicitly: a remote
	API client or mock is not evidence of executable GPU data parallelism.
	"""
	import copy

	devs = [str(device) for device in devices if str(device)]
	if not devs:
		raise ValueError("embedding replicas require a non-empty device list")
	if len(set(devs)) != len(devs):
		raise ValueError(f"embedding replica devices must be unique, got {devs!r}")

	def _build_one(device: str):
		if isinstance(config, str):
			try:
				lazy = embedding_models[config]
			except KeyError as exc:
				raise ValueError(
					f"Embedding model {config!r} is not supported"
				) from exc
			if (
				_HF_EMBEDDING_CLS is None
				or not isinstance(lazy, LazyInit)
				or lazy._factory is not _HF_EMBEDDING_CLS
			):
				raise NotImplementedError(
					"multi-GPU semantic retrieval requires a local HuggingFace "
					f"embedding model, got {config!r}"
				)
			return lazy._factory(
				*lazy._args,
				**{**lazy._kwargs, "device": device},
			)

		option = copy.deepcopy(config)
		if isinstance(option, list):
			if len(option) != 1:
				raise ValueError("Only one embedding model is supported")
			option = option[0]
		if not isinstance(option, dict) or option.get("type") != "huggingface":
			raise NotImplementedError(
				"multi-GPU semantic retrieval requires a local HuggingFace "
				f"embedding model, got {config!r}"
			)
		option["device"] = device
		return EmbeddingModel.load(
			option, embedding_dim=embedding_dim,
		)()

	replicas: List[object] = []
	start = 0
	if first_replica is not None:
		replicas.append(first_replica)
		start = 1
	for device in devs[start:]:
		replicas.append(_build_one(device))
	return replicas


# Maps model factory classes to their dimension kwarg name.
# Models in this dict support configurable output dimensions.
_CONFIGURABLE_DIM_MODELS = {
	OpenAIEmbedding: "dimensions",
	OpenAILikeEmbedding: "dimensions",
	MockEmbeddingRandom: "embed_dim",
}


class _ValidatedLazyInit(LazyInit):
	"""LazyInit wrapper that validates embedding dimension on first use.

	Used for models with fixed output dimensions (e.g. HuggingFace) where the
	dimension cannot be configured but must match the expected value.
	"""

	def __init__(self, inner: LazyInit, expected_dim: int):
		self._inner = inner
		self._expected_dim = expected_dim
		self._instance = None

	def __call__(self):
		if self._instance is None:
			self._instance = self._inner()
			actual = len(self._instance.get_text_embedding("dim_check"))
			if actual != self._expected_dim:
				raise ValueError(
					f"embedding_dim={self._expected_dim} but model "
					f"'{type(self._instance).__name__}' outputs {actual}-dim vectors"
				)
		return self._instance

	def __getattr__(self, name):
		if self._instance is None:
			self()
		return getattr(self._instance, name)


class EmbeddingModel:
	@staticmethod
	def load(config: Union[str, Dict, List[Dict]], embedding_dim: Optional[int] = None):
		if isinstance(config, str):
			return EmbeddingModel.load_from_str(config, embedding_dim=embedding_dim)
		elif isinstance(config, dict):
			return EmbeddingModel.load_from_dict(config, embedding_dim=embedding_dim)
		elif isinstance(config, list):
			return EmbeddingModel.load_from_list(config, embedding_dim=embedding_dim)
		else:
			raise ValueError("Invalid type of config")

	@staticmethod
	def load_from_str(name: str, embedding_dim: Optional[int] = None):
		try:
			lazy_init = embedding_models[name]
		except KeyError:
			raise ValueError(f"Embedding model '{name}' is not supported")

		if embedding_dim is not None:
			dim_kwarg = _CONFIGURABLE_DIM_MODELS.get(lazy_init._factory)
			if dim_kwarg:
				return LazyInit(
					lazy_init._factory, *lazy_init._args,
					**{**lazy_init._kwargs, dim_kwarg: embedding_dim},
				)
			return _ValidatedLazyInit(lazy_init, expected_dim=embedding_dim)

		return lazy_init

	@staticmethod
	def load_from_list(option: List[dict], embedding_dim: Optional[int] = None):
		if len(option) != 1:
			raise ValueError("Only one embedding model is supported")
		return EmbeddingModel.load_from_dict(option[0], embedding_dim=embedding_dim)

	@staticmethod
	def load_from_dict(option: dict, embedding_dim: Optional[int] = None):
		def _check_keys(target: dict):
			if "type" not in target or "model_name" not in target:
				raise ValueError("Both 'type' and 'model_name' must be provided")
			if target["type"] not in [
				"openai",
				"huggingface",
				"mock",
				"ollama",
				"vllm",
			]:
				raise ValueError(
					f"Embedding model type '{target['type']}' is not supported"
				)

		def _get_huggingface_class():
			module = sys.modules.get("llama_index.embeddings.huggingface")
			if not module:
				logger.info(
					"You are using API version of RAG-Stack. "
					"To use local version, run `pip install '.[gpu]'`."
				)
				return None
			return getattr(module, "HuggingFaceEmbedding", None)

		_check_keys(option)

		model_options = option
		model_type = model_options.pop("type")

		embedding_map = {
			"openai": OpenAIEmbedding,
			"mock": MockEmbeddingRandom,
			"huggingface": _get_huggingface_class(),
			"ollama": OllamaEmbedding,
			"openai_like": OpenAILikeEmbedding,
			"vllm": VllmEmbedding,
		}

		embedding_class = embedding_map.get(model_type)
		if not embedding_class:
			raise ValueError(f"Embedding model type '{model_type}' is not supported")

		lazy = LazyInit(embedding_class, **model_options)

		if embedding_dim is not None:
			dim_kwarg = _CONFIGURABLE_DIM_MODELS.get(embedding_class)
			if dim_kwarg:
				return LazyInit(embedding_class, **{**model_options, dim_kwarg: embedding_dim})
			return _ValidatedLazyInit(lazy, expected_dim=embedding_dim)

		return lazy
