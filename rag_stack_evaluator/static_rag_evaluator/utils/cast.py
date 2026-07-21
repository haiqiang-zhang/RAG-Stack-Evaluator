# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

# EXACT unsuffixed result-column names a retrieval-side node overwrites. The
# `_semantic` / `_lexical` provenance variants are deliberately NOT listed: they
# carry pre-rerank / per-arm retriever output that downstream consumers depend
# on (retrieval_token_* metrics + pre-rerank retrieval_recall read
# `retrieved_*_semantic`; the hybrid-fusion node asserts both `_semantic` and
# `_lexical`). See static_rag_evaluator._create_metric_inputs.
_RETRIEVAL_RESULT_COLUMNS = frozenset(
	("retrieved_contents", "retrieved_ids", "retrieve_scores")
)


def drop_retrieval_columns(df):
	"""Drop ONLY the unsuffixed retrieval result columns, preserving the
	`_semantic` / `_lexical` provenance variants.

	A reranker / filter / augmenter calls this right before
	``pd.concat([previous_result, result], axis=1)`` to emit its own fresh
	unsuffixed ``retrieved_contents`` — dropping the stale unsuffixed columns
	avoids a duplicate-column collision. The suffixed provenance columns never
	collide (the node emits unsuffixed names only), so blanket-dropping them by
	prefix was wrong: it erased ``retrieved_contents_semantic`` before the final
	result, silently nulling ``retrieval_token_recall`` / ``_precision`` (and
	demoting pre-rerank ``retrieval_recall`` to post-rerank IDs).
	"""
	cols = [c for c in df.columns if c in _RETRIEVAL_RESULT_COLUMNS]
	return df.drop(columns=cols)


def cast_retrieve_infos(previous_result):
	return {
		"retrieved_contents": cast_retrieved_contents(previous_result),
		"retrieved_ids": cast_retrieved_ids(previous_result),
		"retrieve_scores": cast_retrieve_scores(previous_result),
	}


def cast_retrieved_contents(previous_result):
	if "retrieved_contents" in previous_result.columns:
		return previous_result["retrieved_contents"].tolist()
	elif "retrieved_contents_semantic" in previous_result.columns:
		return previous_result["retrieved_contents_semantic"].tolist()
	elif "retrieved_contents_lexical" in previous_result.columns:
		return previous_result["retrieved_contents_lexical"].tolist()
	else:
		raise ValueError(
			"previous_result must contain either 'retrieved_contents', 'retrieved_contents_semantic', or 'retrieved_contents_lexical' columns."
		)


def cast_retrieved_ids(previous_result):
	if "retrieved_ids" in previous_result.columns:
		return previous_result["retrieved_ids"].tolist()
	elif "retrieved_ids_semantic" in previous_result.columns:
		return previous_result["retrieved_ids_semantic"].tolist()
	elif "retrieved_ids_lexical" in previous_result.columns:
		return previous_result["retrieved_ids_lexical"].tolist()
	else:
		raise ValueError(
			"previous_result must contain either 'retrieved_ids', 'retrieved_ids_semantic', or 'retrieved_ids_lexical' columns."
		)


def cast_retrieve_scores(previous_result):
	if "retrieve_scores" in previous_result.columns:
		return previous_result["retrieve_scores"].tolist()
	elif "retrieve_scores_semantic" in previous_result.columns:
		return previous_result["retrieve_scores_semantic"].tolist()
	elif "retrieve_scores_lexical" in previous_result.columns:
		return previous_result["retrieve_scores_lexical"].tolist()
	else:
		raise ValueError(
			"previous_result must contain either 'retrieve_scores', 'retrieve_scores_semantic', or 'retrieve_scores_lexical' columns."
		)
