import importlib
import re
from typing import Callable, Dict, List


def dynamically_find_function(key: str, target_dict: Dict) -> Callable:
	if key in target_dict:
		module_path, func_name = target_dict[key]
		module = importlib.import_module(module_path)
		func = getattr(module, func_name)
		return func
	else:
		raise KeyError(f"Input module or node {key} is not supported.")


# ---------------------------------------------------------------------------
# Module registry — maps component (or PascalCase alias) → (import_path, class_name)
# ---------------------------------------------------------------------------
SUPPORT_MODULES = {
	# parse
	"langchain_parse": ("rag_stack.dataset_generator.given_raw_article.parse", "langchain_parse"),
	"clova": ("rag_stack.dataset_generator.given_raw_article.parse.clova", "clova_ocr"),
	"llamaparse": ("rag_stack.dataset_generator.given_raw_article.parse.llamaparse", "llama_parse"),
	"table_hybrid_parse": (
		"rag_stack.dataset_generator.given_raw_article.parse.table_hybrid_parse",
		"table_hybrid_parse",
	),
	# chunk
	"llama_index_chunk": ("rag_stack.static_rag_evaluator.chunk", "llama_index_chunk"),
	"langchain_chunk": ("rag_stack.static_rag_evaluator.chunk", "langchain_chunk"),
	# query_expansion
	"query_decompose": ("rag_stack.static_rag_evaluator.nodes.queryexpansion", "QueryDecompose"),
	"hyde": ("rag_stack.static_rag_evaluator.nodes.queryexpansion", "HyDE"),
	"multi_query_expansion": (
		"rag_stack.static_rag_evaluator.nodes.queryexpansion",
		"MultiQueryExpansion",
	),
	"QueryDecompose": ("rag_stack.static_rag_evaluator.nodes.queryexpansion", "QueryDecompose"),
	"HyDE": ("rag_stack.static_rag_evaluator.nodes.queryexpansion", "HyDE"),
	"MultiQueryExpansion": (
		"rag_stack.static_rag_evaluator.nodes.queryexpansion",
		"MultiQueryExpansion",
	),
	# retrieval
	"bm25": ("rag_stack.static_rag_evaluator.nodes.lexicalretrieval", "BM25"),
	"BM25": ("rag_stack.static_rag_evaluator.nodes.lexicalretrieval", "BM25"),
	"vectordb": ("rag_stack.static_rag_evaluator.nodes.semanticretrieval", "VectorDB"),
	"VectorDB": ("rag_stack.static_rag_evaluator.nodes.semanticretrieval", "VectorDB"),
	"hybrid_rrf": ("rag_stack.static_rag_evaluator.nodes.hybridretrieval", "HybridRRF"),
	"HybridRRF": ("rag_stack.static_rag_evaluator.nodes.hybridretrieval", "HybridRRF"),
	"hybrid_cc": ("rag_stack.static_rag_evaluator.nodes.hybridretrieval", "HybridCC"),
	"HybridCC": ("rag_stack.static_rag_evaluator.nodes.hybridretrieval", "HybridCC"),
	# passage_augmenter
	"prev_next_augmenter": (
		"rag_stack.static_rag_evaluator.nodes.passageaugmenter",
		"PrevNextPassageAugmenter",
	),
	"PrevNextPassageAugmenter": (
		"rag_stack.static_rag_evaluator.nodes.passageaugmenter",
		"PrevNextPassageAugmenter",
	),
	# passage_reranker
	"monot5": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "MonoT5"),
	"MonoT5": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "MonoT5"),
	"tart": ("rag_stack.static_rag_evaluator.nodes.passagereranker.tart", "Tart"),
	"Tart": ("rag_stack.static_rag_evaluator.nodes.passagereranker.tart", "Tart"),
	"upr": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "Upr"),
	"Upr": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "Upr"),
	"koreranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "KoReranker"),
	"KoReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "KoReranker"),
	"cohere_reranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "CohereReranker"),
	"CohereReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "CohereReranker"),
	"rankgpt": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "RankGPT"),
	"RankGPT": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "RankGPT"),
	"jina_reranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "JinaReranker"),
	"JinaReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "JinaReranker"),
	"colbert_reranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "ColbertReranker"),
	"ColbertReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "ColbertReranker"),
	"sentence_transformer_reranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"SentenceTransformerReranker",
	),
	"SentenceTransformerReranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"SentenceTransformerReranker",
	),
	"flag_embedding_reranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"FlagEmbeddingReranker",
	),
	"FlagEmbeddingReranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"FlagEmbeddingReranker",
	),
	"flag_embedding_llm_reranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"FlagEmbeddingLLMReranker",
	),
	"FlagEmbeddingLLMReranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"FlagEmbeddingLLMReranker",
	),
	"time_reranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "TimeReranker"),
	"TimeReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "TimeReranker"),
	"openvino_reranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "OpenVINOReranker"),
	"OpenVINOReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "OpenVINOReranker"),
	"voyageai_reranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "VoyageAIReranker"),
	"VoyageAIReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "VoyageAIReranker"),
	"mixedbreadai_reranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"MixedbreadAIReranker",
	),
	"MixedbreadAIReranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker",
		"MixedbreadAIReranker",
	),
	"flashrank_reranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "FlashRankReranker"),
	"FlashRankReranker": ("rag_stack.static_rag_evaluator.nodes.passagereranker", "FlashRankReranker"),
	# passage_filter
	"similarity_threshold_cutoff": (
		"rag_stack.static_rag_evaluator.nodes.passagefilter",
		"SimilarityThresholdCutoff",
	),
	"similarity_percentile_cutoff": (
		"rag_stack.static_rag_evaluator.nodes.passagefilter",
		"SimilarityPercentileCutoff",
	),
	"recency_filter": ("rag_stack.static_rag_evaluator.nodes.passagefilter", "RecencyFilter"),
	"threshold_cutoff": ("rag_stack.static_rag_evaluator.nodes.passagefilter", "ThresholdCutoff"),
	"percentile_cutoff": ("rag_stack.static_rag_evaluator.nodes.passagefilter", "PercentileCutoff"),
	"SimilarityThresholdCutoff": (
		"rag_stack.static_rag_evaluator.nodes.passagefilter",
		"SimilarityThresholdCutoff",
	),
	"SimilarityPercentileCutoff": (
		"rag_stack.static_rag_evaluator.nodes.passagefilter",
		"SimilarityPercentileCutoff",
	),
	"RecencyFilter": ("rag_stack.static_rag_evaluator.nodes.passagefilter", "RecencyFilter"),
	"ThresholdCutoff": ("rag_stack.static_rag_evaluator.nodes.passagefilter", "ThresholdCutoff"),
	"PercentileCutoff": ("rag_stack.static_rag_evaluator.nodes.passagefilter", "PercentileCutoff"),
	# passage_compressor
	"tree_summarize": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "TreeSummarize"),
	"refine": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "Refine"),
	"longllmlingua": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "LongLLMLingua"),
	"llmlingua2": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "LLMLingua2"),
	"TreeSummarize": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "TreeSummarize"),
	"Refine": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "Refine"),
	"LongLLMLingua": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "LongLLMLingua"),
	"LLMLingua2": ("rag_stack.static_rag_evaluator.nodes.passagecompressor", "LLMLingua2"),
	# prompt_maker
	"fstring": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "Fstring"),
	"long_context_reorder": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "LongContextReorder"),
	"window_replacement": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "WindowReplacement"),
	"Fstring": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "Fstring"),
	"LongContextReorder": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "LongContextReorder"),
	"WindowReplacement": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "WindowReplacement"),
	"chat_fstring": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "ChatFstring"),
	"ChatFstring": ("rag_stack.static_rag_evaluator.nodes.promptmaker", "ChatFstring"),
	# generator — backends are now in nodes/generator/registry.py (GENERATOR_BACKENDS)
	# These entries remain for the generator node runner which uses SUPPORT_MODULES
	"llama_index_llm": ("rag_stack.static_rag_evaluator.nodes.generator", "LlamaIndexLLM"),
	"vllm": ("rag_stack.static_rag_evaluator.nodes.generator", "Vllm"),
	"openai_llm": ("rag_stack.static_rag_evaluator.nodes.generator", "OpenAILLM"),
	"vllm_api": ("rag_stack.static_rag_evaluator.nodes.generator", "VllmAPI"),
	"LlamaIndexLLM": ("rag_stack.static_rag_evaluator.nodes.generator", "LlamaIndexLLM"),
	"Vllm": ("rag_stack.static_rag_evaluator.nodes.generator", "Vllm"),
	"OpenAILLM": ("rag_stack.static_rag_evaluator.nodes.generator", "OpenAILLM"),
	"VllmAPI": ("rag_stack.static_rag_evaluator.nodes.generator", "VllmAPI"),
	# dataset_generator
	"given_raw_article": (
		"rag_stack.dataset_generator.given_raw_article.generator",
		"GivenRawArticleGenerator",
	),
	"synthetic_article": (
		"rag_stack.dataset_generator.synthetic_article.generator",
		"SyntheticArticleGenerator",
	),
}

# ---------------------------------------------------------------------------
# Node registry — maps stage → (import_path, runner_function_name)
# ---------------------------------------------------------------------------
SUPPORT_NODES = {
	"query_expansion": (
		"rag_stack.static_rag_evaluator.nodes.queryexpansion.run",
		"run_query_expansion_node",
	),
	"semantic_retrieval": (
		"rag_stack.static_rag_evaluator.nodes.semanticretrieval.run",
		"run_semantic_retrieval_node",
	),
	"dense_retrieval": (
		"rag_stack.static_rag_evaluator.nodes.semanticretrieval.run",
		"run_semantic_retrieval_node",
	),
	"lexical_retrieval": (
		"rag_stack.static_rag_evaluator.nodes.lexicalretrieval.run",
		"run_lexical_retrieval_node",
	),
	"sparse_retrieval": (
		"rag_stack.static_rag_evaluator.nodes.lexicalretrieval.run",
		"run_lexical_retrieval_node",
	),
	"hybrid_retrieval": (
		"rag_stack.static_rag_evaluator.nodes.hybridretrieval.run",
		"run_hybrid_retrieval_node",
	),
	"generator": ("rag_stack.static_rag_evaluator.nodes.generator.run", "run_generator_node"),
	"prompt_maker": ("rag_stack.static_rag_evaluator.nodes.promptmaker.run", "run_prompt_maker_node"),
	"passage_filter": (
		"rag_stack.static_rag_evaluator.nodes.passagefilter.run",
		"run_passage_filter_node",
	),
	"passage_compressor": (
		"rag_stack.static_rag_evaluator.nodes.passagecompressor.run",
		"run_passage_compressor_node",
	),
	"passage_reranker": (
		"rag_stack.static_rag_evaluator.nodes.passagereranker.run",
		"run_passage_reranker_node",
	),
	"passage_augmenter": (
		"rag_stack.static_rag_evaluator.nodes.passageaugmenter.run",
		"run_passage_augmenter_node",
	),
}


def get_support_modules(module_name: str) -> Callable:
	return dynamically_find_function(module_name, SUPPORT_MODULES)


def get_support_nodes(node_name: str) -> Callable:
	return dynamically_find_function(node_name, SUPPORT_NODES)


# ---------------------------------------------------------------------------
# Registry introspection — derive stage→modules and vectordb_types
# directly from the module-level dicts above.
# ---------------------------------------------------------------------------

def get_stage_components() -> Dict[str, List[str]]:
	"""Derive {stage: [component, ...]} from SUPPORT_NODES and SUPPORT_MODULES.

	Uses SUPPORT_NODES to get canonical stage names and their import paths,
	then matches modules from SUPPORT_MODULES by their import path segment.
	Only includes lowercase keys (skips PascalCase aliases).
	"""
	# Build path_segment → stage from SUPPORT_NODES
	path_to_type: Dict[str, str] = {}
	for stage, (module_path, _) in SUPPORT_NODES.items():
		match = re.search(r"\.nodes\.(\w+)", module_path)
		if match:
			path_to_type[match.group(1).split(".")[0]] = stage

	# Group modules by stage
	node_modules: Dict[str, List[str]] = {nt: [] for nt in path_to_type.values()}
	for key, (module_path, _) in SUPPORT_MODULES.items():
		if key != key.lower():
			continue
		match = re.search(r"\.nodes\.(\w+)", module_path)
		if not match:
			continue
		path_seg = match.group(1).split(".")[0]
		stage = path_to_type.get(path_seg)
		if stage:
			node_modules[stage].append(key)

	return node_modules


def get_vectordb_types() -> List[str]:
	"""Extract lowercase vectordb type names from the vectordb registry."""
	from rag_stack.static_rag_evaluator.vectordb import SUPPORT_VECTORDB
	return sorted(set(k for k in SUPPORT_VECTORDB if k == k.lower()))
