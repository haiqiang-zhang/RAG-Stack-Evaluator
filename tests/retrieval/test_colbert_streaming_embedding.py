from types import SimpleNamespace

import torch

from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.colbert import (
    get_colbert_embedding_batch,
)


class _Tokenizer:
    def __call__(self, input_strings, **kwargs):
        del kwargs
        rows = len(input_strings)
        values = torch.arange(rows * 3, dtype=torch.long).reshape(rows, 3)
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
            "token_type_ids": torch.zeros_like(values),
        }


class _Model:
    config = SimpleNamespace(max_position_embeddings=3)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        del attention_mask, token_type_ids
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 2)
        return SimpleNamespace(last_hidden_state=hidden)


def test_colbert_embedding_streams_microbatches_to_cpu(monkeypatch):
    def _full_cohort_cat_is_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ColBERT must not concatenate the full cohort on GPU")

    monkeypatch.setattr(torch, "cat", _full_cohort_cat_is_forbidden)

    embeddings = get_colbert_embedding_batch(
        ["q0", "q1", "q2", "q3", "q4"],
        _Model(),
        _Tokenizer(),
        batch_size=2,
        device="cpu",
    )

    assert len(embeddings) == 5
    assert [embedding.shape for embedding in embeddings] == [(1, 3, 2)] * 5
    assert embeddings[0][0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert embeddings[-1][0, :, 0].tolist() == [12.0, 13.0, 14.0]
