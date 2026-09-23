"""rag-stack-owned FlashRAG pipelines for methods the vendored fork lacks.

Two agentic methods are implemented on top of FlashRAG primitives instead
of patching large new classes into the vendored package:

  * :class:`SearchO1Pipeline` — Search-o1 (agentic reason-and-search with
    the paper's Reason-in-Documents step): subclass of the fork's
    parameterized :class:`ReasoningPipeline` loop with Search-o1 prompt /
    tokens, overriding ``_build_doc_strings`` so every retrieval round runs
    an extra batched generator pass that distills the raw documents w.r.t.
    the in-flight search query before insertion.
  * :class:`ReActPipeline` — classic ReAct (Thought → Action → Observation
    loop with ``Search[...]`` / ``Finish[...]`` actions), a self-contained
    frontier loop modeled on ReasoningPipeline.run.

Both wrap every generator/retriever call site in ``query_context`` so the
monitor attributes component calls to per-query trace_v1 payloads — the
trace-driven cost model needs no method-specific handling (trace stays
pipeline-shape-agnostic).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from flashrag.monitor_hook import query_context

logger = logging.getLogger("RAG-Stack")
from flashrag.pipeline import BasicPipeline
from flashrag.pipeline.reasoning_pipeline import ReasoningPipeline
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator, get_retriever


class SearchO1Pipeline(ReasoningPipeline):
    """Search-o1: o1-style long reasoning with on-demand search and the
    Reason-in-Documents refinement (arXiv:2501.05366).

    Differences from the base ReasoningPipeline (= Search-R1 style):
      * Search-o1 token set (``<|begin_search_query|>`` …).
      * Reason-in-Documents: retrieved documents are NOT inserted raw — a
        batched auxiliary generation distills them w.r.t. the current
        search query, and only the distilled text enters the reasoning
        context. This is the method's signature extra generate call per
        retrieval round (visible in the trace, priced by the cost model).
    """

    system_prompt = ""
    # Includes a compact worked example: small instruct models rarely emit
    # the search tokens zero-shot (they answer directly and the loop
    # degenerates to a single generate with no retrieval).
    user_prompt = (
        "You are a reasoning assistant. Solve the question by thinking step "
        "by step, and search the knowledge base whenever you are missing "
        "knowledge.\n"
        "To search: write <|begin_search_query|> your query (keywords) "
        "<|end_search_query|> and stop; the system will reply with "
        "<|begin_search_result|> ...helpful information... "
        "<|end_search_result|> and you continue reasoning.\n"
        "When you know the answer, write it enclosed in <answer> </answer> "
        "tags.\n\n"
        "Example:\n"
        "Question: Where was the author of \"Walden\" born?\n"
        "\"Walden\" was written by Henry David Thoreau; I need his "
        "birthplace.\n"
        "<|begin_search_query|> Henry David Thoreau birthplace "
        "<|end_search_query|>\n"
        "<|begin_search_result|> Thoreau was born in Concord, Massachusetts "
        "in 1817. <|end_search_result|>\n"
        "So the answer is <answer> Concord, Massachusetts </answer>\n\n"
        "Now solve the real question the same way. Always search before "
        "answering.\n"
        "Question: {question}\n"
    )

    _RID_PROMPT = (
        "You are analyzing retrieved documents inside an ongoing reasoning "
        "process.\n"
        "Current search query: {query}\n"
        "Retrieved documents:\n{docs}\n\n"
        "Extract ONLY the factual information from the documents that helps "
        "answer the search query, in at most three sentences. If the "
        "documents contain no helpful information, output exactly: "
        "No helpful information found.\n"
        "Helpful information:"
    )

    def __init__(
        self,
        config,
        prompt_template=None,
        max_retrieval_num=5,
        reason_in_documents=True,
        retriever=None,
        generator=None,
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
        super().__init__(
            config,
            prompt_template=prompt_template,
            max_retrieval_num=max_retrieval_num,
            begin_of_query_token="<|begin_search_query|>",
            end_of_query_token="<|end_search_query|>",
            begin_of_documents_token="<|begin_search_result|>",
            end_of_documents_token="<|end_search_result|>",
            retriever=retriever,
            generator=generator,
        )
        self.reason_in_documents = bool(reason_in_documents)

    def _build_doc_strings(self, step_query_list, retrieved_docs, current_step_idx):
        if not self.reason_in_documents:
            return super()._build_doc_strings(
                step_query_list, retrieved_docs, current_step_idx
            )
        rid_prompts = []
        for it, docs in zip(step_query_list, retrieved_docs):
            # Cap raw passages so the refinement prompt itself fits the
            # context window on large-chunk corpora.
            raw = "\n".join(
                f"({i + 1}) {str(doc['contents'])[:800]}"
                for i, doc in enumerate(docs)
            )
            rid_prompts.append(
                self._RID_PROMPT.format(query=it["query"], docs=raw)
            )
        # Reason-in-Documents distillation — an additional generate call for
        # the same items/step, attributed in the trace like any other call.
        with query_context(
            [str(it["item"].id) for it in step_query_list],
            step_idx=current_step_idx,
        ):
            refined = self.generator.generate(rid_prompts)
        return [
            f"\n\n{self.begin_of_documents_token}\n{r.strip()}\n"
            f"{self.end_of_documents_token}\n\n"
            for r in refined
        ]


class ReActPipeline(BasicPipeline):
    """ReAct (arXiv:2210.03629): Thought → Action → Observation loop.

    Actions: ``Search[query]`` (retrieve passages, fed back as an
    Observation) and ``Finish[answer]`` (terminate). Items run as a
    frontier batch like ReasoningPipeline: one generate per round over the
    unfinished items, one batched retrieval for the round's Search actions.
    """

    system_prompt = ""
    # One-shot exemplar: small instruct models do not follow the ReAct format
    # zero-shot (they answer directly and never emit Search[...]) — the
    # worked example locks the Thought/Action/Observation syntax.
    user_prompt = (
        "Solve the question by interleaving Thought, Action and Observation "
        "steps.\n"
        "Thought reasons about the current situation. Action must be exactly "
        "one of:\n"
        "(1) Search[query] — search the knowledge base for `query`.\n"
        "(2) Finish[answer] — conclude with `answer` as the final answer.\n"
        "After every Search action the system supplies an Observation with "
        "retrieved passages. Always Search before answering; Finish as soon "
        "as you can answer.\n\n"
        "Example:\n"
        "Question: Where was the author of \"Walden\" born?\n"
        "Thought 1: \"Walden\" was written by Henry David Thoreau. I need "
        "his birthplace.\n"
        "Action 1: Search[Henry David Thoreau birthplace]\n"
        "Observation 1: (1) Henry David Thoreau: born July 12, 1817, in "
        "Concord, Massachusetts, ...\n"
        "Thought 2: The observation says Thoreau was born in Concord, "
        "Massachusetts.\n"
        "Action 2: Finish[Concord, Massachusetts]\n\n"
        "Now solve the real question the same way.\n"
        "Question: {question}\n"
        "Thought 1:"
    )

    _SEARCH_RE = re.compile(r"Search\s*\[(.+?)\]", re.DOTALL | re.IGNORECASE)
    _FINISH_RE = re.compile(r"Finish\s*\[(.+?)\]", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        config,
        prompt_template=None,
        max_iter=5,
        retriever=None,
        generator=None,
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
        super().__init__(config, prompt_template)
        self.generator = generator if generator is not None else get_generator(config)
        self.retriever = retriever if retriever is not None else get_retriever(config)
        self.max_iter = int(max_iter)
        # Stop BEFORE the model hallucinates its own Observation.
        self.stop_tokens = ["Observation", "<|im_end|>", "<|endoftext|>"]

    # ReAct observations are snippets, not full passages: corpora with large
    # chunks (dragonball ~1.7k tokens/chunk) would blow the context window
    # within one or two rounds otherwise.
    _OBS_CHARS_PER_DOC = 500

    @classmethod
    def _docs_to_observation(cls, docs: List[Dict]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            contents = str(doc.get("contents", ""))
            title = contents.split("\n")[0]
            text = " ".join(contents.split("\n")[1:]) or title
            if len(text) > cls._OBS_CHARS_PER_DOC:
                text = text[: cls._OBS_CHARS_PER_DOC] + "…"
            parts.append(f"({i + 1}) {title}: {text}")
        return " ".join(parts) if parts else "No results found."

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        prompts = [
            self.prompt_template.get_string(question=q) for q in dataset.question
        ]
        dataset.update_output("prompt", prompts)
        dataset.update_output("finish_flag", [False] * len(prompts))
        dataset.update_output("react_round", [1] * len(prompts))
        dataset.update_output("retrieval_results", [{} for _ in range(len(prompts))])
        dataset.update_output("retrieved_times", [0] * len(prompts))

        for step_idx in range(self.max_iter + 1):
            frontier = [item for item in dataset if not item.finish_flag]
            if not frontier:
                break
            if step_idx == self.max_iter:
                for item in frontier:
                    item.pred = "No valid answer found"
                    item.finish_flag = True
                    item.finish_reason = "Reach max iterations"
                break

            # Middle-truncate accumulated prompts to the generator's input
            # budget (keeps the head instructions + the freshest thoughts).
            with query_context(
                [str(item.id) for item in frontier], step_idx=step_idx
            ):
                outputs = self.generator.generate(
                    [
                        self.prompt_template.truncate_prompt(item.prompt)
                        for item in frontier
                    ],
                    stop=self.stop_tokens,
                )
            if step_idx == 0 and outputs:
                # One sample per run: the cheapest way to see whether the
                # model actually follows the ReAct format (small models
                # often don't — that shows up here, not as an exception).
                logger.info(f"[react] round-0 sample output: {outputs[0][:400]!r}")

            searches = []  # [{'item':, 'query':}]
            for item, out in zip(frontier, outputs):
                out = out.strip()
                item.prompt = item.prompt + " " + out
                finish_m = self._FINISH_RE.findall(out)
                search_m = self._SEARCH_RE.findall(out)
                if finish_m:
                    item.pred = finish_m[-1].strip()
                    item.finish_flag = True
                    item.finish_reason = "Finished"
                elif search_m:
                    searches.append({"item": item, "query": search_m[-1].strip()})
                else:
                    # No parseable action — treat the trailing thought as the
                    # answer (mirrors ReasoningPipeline's 'normal finish').
                    item.pred = out
                    item.finish_flag = True
                    item.finish_reason = "Normal finish without action"

            if searches:
                with query_context(
                    [str(s["item"].id) for s in searches], step_idx=step_idx
                ):
                    docs_per_query = self.retriever.batch_search(
                        [s["query"] for s in searches]
                    )
                for s, docs in zip(searches, docs_per_query):
                    item = s["item"]
                    item.retrieval_results[item.retrieved_times] = {
                        "query": s["query"],
                        "docs": list(docs),
                    }
                    item.retrieved_times += 1
                    n = item.react_round
                    item.prompt += (
                        f"\nObservation {n}: {self._docs_to_observation(docs)}"
                        f"\nThought {n + 1}:"
                    )
                    item.react_round = n + 1

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )
        return dataset
