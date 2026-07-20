"""ReAct agentic loop for the AutoRAG (static) evaluator.

The static evaluator's normal path is a sequential node_line run. ReAct is the
first **agentic** method on this backend: a Thought → Action → Observation loop
where the LLM itself decides when to retrieve (``Search[query]``) and when to
answer (``Finish[answer]``). It reuses AutoRAG's own retriever
(:class:`~rag_stack.static_rag_evaluator.nodes.semanticretrieval.vectordb.VectorDB`)
and generator (:class:`~rag_stack.static_rag_evaluator.nodes.generator.vllm.Vllm`)
— no FlashRAG dependency.

It emits a ``trace_v1`` payload (one ordered component-call list per query) so
the trace-driven cost model (:class:`rag_stack.cost_model.assembly.RAGCMAssembly`) prices it exactly
like any other agentic pipeline — the data-dependent step count lives in the
trace, the cost model never inspects the method. ``Read``-style chunk tools are
deliberately out of scope here (this is the *simplest* agentic method); add them
later the same way A-RAG did on the FlashRAG side.

Measured ReAct serving is implemented in
``rag_stack.static_rag_evaluator.measured.serving_runtime``. This module is the
quality/trace loop only.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rag_stack.static_rag_evaluator.utils.util import fetch_contents

logger = logging.getLogger("RAG-Stack")

# --- ReAct prompt (one-shot exemplar) --------------------------------------
# Small instruct models do not follow the Thought/Action/Observation format
# zero-shot — the worked example locks the syntax and the Search-before-answer
# habit (mirrors the FlashRAG-side ReAct prompt).
_REACT_PROMPT = (
    "Solve the question by interleaving Thought, Action and Observation steps.\n"
    "Thought reasons about the current situation. Action must be exactly one of:\n"
    "(1) Search[query] — search the knowledge base for `query`.\n"
    "(2) Finish[answer] — conclude with `answer` as the final answer.\n"
    "After every Search action the system supplies an Observation with retrieved "
    "passages. Always Search before answering; Finish as soon as you can answer.\n\n"
    "Example:\n"
    "Question: Where was the author of \"Walden\" born?\n"
    "Thought 1: \"Walden\" was written by Henry David Thoreau. I need his birthplace.\n"
    "Action 1: Search[Henry David Thoreau birthplace]\n"
    "Observation 1: (1) Henry David Thoreau: born July 12, 1817, in Concord, "
    "Massachusetts, ...\n"
    "Thought 2: The observation says Thoreau was born in Concord, Massachusetts.\n"
    "Action 2: Finish[Concord, Massachusetts]\n\n"
    "Now solve the real question the same way.\n"
    "Question: {question}\n"
    "Thought 1:"
)

_SEARCH_RE = re.compile(r"Search\s*\[(.+?)\]", re.DOTALL | re.IGNORECASE)
_FINISH_RE = re.compile(r"Finish\s*\[(.+?)\]", re.DOTALL | re.IGNORECASE)
_STOP_TOKENS = ["Observation", "<|im_end|>", "<|endoftext|>"]
# Observation truncation: dragonball chunks are ~1.7k tokens — without this the
# context window blows within a couple of rounds.
_OBS_CHARS_PER_DOC = 500


def _docs_to_observation(contents: List[str]) -> str:
    parts = []
    for i, contents_i in enumerate(contents):
        text = str(contents_i)
        title = text.split("\n")[0]
        body = " ".join(text.split("\n")[1:]) or title
        if len(body) > _OBS_CHARS_PER_DOC:
            body = body[:_OBS_CHARS_PER_DOC] + "…"
        parts.append(f"({i + 1}) {title}: {body}")
    return " ".join(parts) if parts else "No results found."


def run_react(
    *,
    qa_data: pd.DataFrame,
    retriever: Any,
    generator: Any,
    generator_model: str,
    embedding_model_id: str,
    top_k: int,
    gen_params: Dict[str, Any],
    reranker: Optional[Any] = None,
    reranker_params: Optional[Dict[str, Any]] = None,
    reranker_model_id: Optional[str] = None,
    nprobe: Optional[int] = None,
    ef_search: Optional[int] = None,
    max_iter: int,
) -> pd.DataFrame:
    """Run the quality/trace-only ReAct loop over ``qa_data``.

    ``result_df`` carries a ``generated_texts`` column (the final answers) so the
    static evaluator's ``_evaluate_final_result`` scores it like any generation;
    calls are recorded into the active recorder for trace construction.
    Deployed-vLLM measured ReAct is handled by ``MeasuredServingRuntime`` and is
    rejected here to prevent the old per-loop timing path from reappearing.

    Args:
        retriever: a ``VectorDB`` instance (``._pure`` + ``.corpus_df``).
        generator: a ``Vllm`` (``BaseGenerator``) instance (``._pure``).
        reranker: optional passage reranker. When present, every Search
            observation is built from reranked candidates.
        generator_model / embedding_model_id: trace ``model_id``s.
        top_k / nprobe / ef_search: retrieval knobs (search-time).
        gen_params: sampling kwargs forwarded to the generator (temperature,
            max_tokens, …); ``stop`` is added here.
        reranker_params / reranker_model_id: optional reranker ``pure`` kwargs
            and trace model id.
        max_iter: THE react round-count knob (react-only sub-config, REQUIRED):
            max generate (Thought/Action) rounds per query. Queries not
            finished at the cap exit truncated; the last allowed round never
            executes a Search (its Observation could not be consumed).
    """
    from rag_stack.static_rag_evaluator import recording as _rec

    questions = [str(q) for q in qa_data["query"].tolist()]
    n = len(questions)
    # Permanent per-query ids — the agentic loop records into the SAME recorder as the
    # sequential nodes (record_io), keyed by these qids. Each round records only the
    # active frontier's qids, so per-query round count M falls out naturally.
    qids = (qa_data["__qid__"].tolist()
            if "__qid__" in qa_data.columns else list(range(n)))

    if getattr(generator, "_subprocess", None) is not None:
        raise RuntimeError(
            "measured ReAct must use MeasuredServingRuntime, not run_react()"
        )

    prompts = [_REACT_PROMPT.format(question=q) for q in questions]
    finished = [False] * n
    preds: List[str] = [""] * n
    rounds = [1] * n
    generate_calls = [0] * n
    retrieval_calls = [0] * n
    truncated = [False] * n
    last_semantic_ids: List[List[str]] = [[] for _ in range(n)]
    last_semantic_contents: List[List[str]] = [[] for _ in range(n)]
    last_semantic_scores: List[List[float]] = [[] for _ in range(n)]
    last_final_ids: List[List[str]] = [[] for _ in range(n)]
    last_final_contents: List[List[str]] = [[] for _ in range(n)]
    last_final_scores: List[List[float]] = [[] for _ in range(n)]

    gen_kwargs = {k: v for k, v in (gen_params or {}).items() if v is not None}
    # `max_iter` is THE react round cap: max Thought/Action/Observation rounds
    # (= generate rounds). It doubles as the dead-loop guard — there is no
    # separate safety constant.
    cap = max(1, int(max_iter))

    for step_idx in range(cap):
        frontier = [i for i in range(n) if not finished[i]]
        if not frontier:
            break

        # --- one generate over the unfinished frontier -----------------------
        batch_prompts = [prompts[i] for i in frontier]
        texts, token_ids, _ = generator._pure(
            batch_prompts, stop=_STOP_TOKENS, **gen_kwargs
        )
        if step_idx == 0 and texts:
            logger.info(f"[react] round-0 sample output: {str(texts[0])[:400]!r}")

        # trace: one "generate" call per frontier query into the shared recorder
        # (output tokens EXACT via token_ids; input = the prompt as sent THIS round).
        _rec.record_io("generator", [qids[i] for i in frontier], list(batch_prompts),
                       out_texts=list(texts), out_token_ids=list(token_ids),
                       model_id=generator_model)
        for i in frontier:
            generate_calls[i] += 1

        searches: List[Tuple[int, str]] = []  # (query_index, search_query)
        for pos, i in enumerate(frontier):
            out = str(texts[pos]).strip()
            prompts[i] = prompts[i] + " " + out

            finish_m = _FINISH_RE.findall(out)
            search_m = _SEARCH_RE.findall(out)
            if finish_m:
                preds[i] = finish_m[-1].strip()
                finished[i] = True
            elif search_m:
                searches.append((i, search_m[-1].strip()))
            else:
                # No parseable action — take the trailing text as the answer.
                preds[i] = out
                finished[i] = True

        # --- one batched retrieval for this round's Search actions -----------
        # On the LAST allowed round the observation could never be consumed by
        # a next generate — skip the search (those queries exit truncated), so
        # the trace never carries a retrieval no generate reads.
        if searches and step_idx < cap - 1:
            for i, _ in searches:
                retrieval_calls[i] += 1
            sq = [[q] for (_, q) in searches]
            ids, scores = retriever._pure(
                sq, top_k=top_k, nprobe=nprobe, ef_search=ef_search
            )
            contents = fetch_contents(retriever.corpus_df, ids)
            semantic_ids = ids
            semantic_scores = scores
            semantic_contents = contents
            _sqids = [qids[i] for (i, _q) in searches]
            _squeries = [q for (_i, q) in searches]
            # trace: encode (query embedding) + retrieve (faiss) for this round's
            # searches — record pre-rerank retrieval output.
            _rec.record_io("semantic_retrieval_encode", _sqids, _squeries, model_id=embedding_model_id)
            _rec.record_io(
                "semantic_retrieval_vectorsearch",
                _sqids,
                _squeries,
                out_texts=contents,
                model_id=embedding_model_id,
            )
            if reranker is not None:
                # The reranker node contract (validate_qa_dataset in
                # cast_to_run) requires the QA identity columns; each search
                # belongs to qa row ``i``, so carry that row's qid /
                # generation_gt. ``query`` is the agent's SEARCH query — the
                # text the candidates are reranked against — not the original
                # question.
                rerank_input = pd.DataFrame(
                    {
                        "qid": [qa_data["qid"].iloc[i] for (i, _q) in searches],
                        "generation_gt": [
                            qa_data["generation_gt"].iloc[i] for (i, _q) in searches
                        ],
                        "query": _squeries,
                        "__qid__": _sqids,
                        "retrieved_contents_semantic": contents,
                        "retrieved_ids_semantic": ids,
                        "retrieve_scores_semantic": scores,
                    }
                )
                reranked = reranker.pure(
                    rerank_input,
                    **dict(reranker_params or {}),
                )
                _rec.record_io(
                    "passage_reranker",
                    _sqids,
                    [
                        [q] + (list(c) if isinstance(c, (list, tuple)) else [c])
                        for q, c in zip(_squeries, contents)
                    ],
                    model_id=reranker_model_id,
                )
                contents = reranked["retrieved_contents"].tolist()
                ids = reranked["retrieved_ids"].tolist()
                scores = reranked["retrieve_scores"].tolist()
            for pos_s, ((i, q), ids_i, contents_i) in enumerate(zip(searches, ids, contents)):
                last_semantic_ids[i] = list(semantic_ids[pos_s])
                last_semantic_contents[i] = list(semantic_contents[pos_s])
                last_semantic_scores[i] = (
                    list(semantic_scores[pos_s]) if pos_s < len(semantic_scores) else []
                )
                last_final_ids[i] = list(ids_i)
                last_final_contents[i] = list(contents_i)
                last_final_scores[i] = list(scores[pos_s]) if pos_s < len(scores) else []
                r = rounds[i]
                prompts[i] += (
                    f"\nObservation {r}: {_docs_to_observation(list(contents_i))}"
                    f"\nThought {r + 1}:"
                )
                rounds[i] = r + 1
    for i in range(n):
        if not finished[i]:
            truncated[i] = True
            finished[i] = True
        # Keep static quality evaluation aligned with measured ReAct: an LLM
        # may emit only whitespace (or ``Finish[   ]``), which is a completed
        # generation but not a usable metric input.  Normalize every empty
        # final prediction, including already-finished queries, without
        # changing their trace or truncation semantics.
        if not str(preds[i]).strip():
            preds[i] = "No valid answer found"

    result = qa_data.copy().reset_index(drop=True)
    result["generated_texts"] = preds
    result["agent_generate_calls"] = generate_calls
    result["agent_retrieval_calls"] = retrieval_calls
    result["agent_truncated"] = truncated
    # Surface the LAST retrieval per query under the semantic-retrieval column
    # names the static evaluator's metric-input builder requires (it asserts
    # retrieved_contents_semantic is present, and reads retrieved_ids_semantic
    # for retrieval metrics). Empty lists for queries that never searched.
    result["retrieved_contents_semantic"] = last_semantic_contents
    result["retrieved_ids_semantic"] = last_semantic_ids
    result["retrieve_scores_semantic"] = last_semantic_scores
    if reranker is not None:
        result["retrieved_contents"] = last_final_contents
        result["retrieved_ids"] = last_final_ids
        result["retrieve_scores"] = last_final_scores

    # The synchronous loop is quality/trace-only. Measured ReAct is globally
    # continuous-batching and therefore must use the async subprocess path above.
    return result
