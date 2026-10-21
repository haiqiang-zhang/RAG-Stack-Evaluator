"""Importable-by-spawn fake stage classes for test_aux_process_ipc.py.

The aux child process imports the stage class by (module, name), so these
fakes must live in a real module — the test adds this directory to the
child's PYTHONPATH.
"""

import pandas as pd

# Columns the parent is expected to ship for passage_reranker (mirrors
# aux_process._SHIP_COLUMNS; asserted inside the child so a pruning
# regression fails the round trip loudly).
ALLOWED_RERANKER_COLUMNS = {
    "qid", "query", "queries", "generation_gt", "__qid__",
    "retrieved_contents", "retrieved_ids", "retrieve_scores",
    "retrieved_contents_semantic", "retrieved_ids_semantic",
    "retrieve_scores_semantic",
    "retrieved_contents_lexical", "retrieved_ids_lexical",
    "retrieve_scores_lexical",
}


class FakeReranker:
    """Permutes + cuts to top-2 by reversed order — id/content pairing kept."""

    def __init__(self, project_dir, **kwargs):
        self.project_dir = project_dir

    def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
        extra = set(previous_result.columns) - ALLOWED_RERANKER_COLUMNS
        if extra:
            raise AssertionError(f"unpruned columns shipped to child: {extra}")
        if "secret_big_column" in previous_result.columns:
            raise AssertionError("pruning failed: big column crossed the pipe")
        contents, ids, scores = [], [], []
        for _, row in previous_result.iterrows():
            order = list(range(len(row["retrieved_ids"])))[::-1][:2]
            contents.append([row["retrieved_contents"][j] for j in order])
            ids.append([row["retrieved_ids"][j] for j in order])
            scores.append([float(10 - j) for j in order])
        return pd.DataFrame(
            {
                "retrieved_contents": contents,
                "retrieved_ids": ids,
                "retrieve_scores": scores,
            }
        )


class FakeRewritingReranker(FakeReranker):
    """Rewrites passage text — the id-keyed reply MUST fall back to full."""

    def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
        out = super().pure(previous_result, *args, **kwargs)
        out["retrieved_contents"] = [
            [c + " REWRITTEN" for c in row] for row in out["retrieved_contents"]
        ]
        return out


class FakeRetrieval:
    """Echo retrieval — returns fixed contents; asserts pruned input."""

    def __init__(self, project_dir, **kwargs):
        self.device = kwargs.get("device")

    def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
        if "retrieved_contents" in previous_result.columns:
            raise AssertionError("retrieval input should not carry contents")
        n = len(previous_result)
        return pd.DataFrame(
            {
                "retrieved_contents_semantic": [["doc"]] * n,
                "retrieved_ids_semantic": [["d0"]] * n,
                "retrieve_scores_semantic": [[1.0]] * n,
            }
        )
