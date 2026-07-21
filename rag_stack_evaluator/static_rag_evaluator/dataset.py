"""Evaluation-side owner of the QA + corpus execution lifecycle.

This module is the evaluator's data plane. It replaces the old
``rag_stack.corpus_qa.CorpusQAManager`` and folds in the chunk cache (now
``rag_stack_evaluator.static_rag_evaluator.chunk_cache``) so that everything dataset-shaped
lives under the static evaluator — you hand it a dataset + config (including the
chunker), and it loads, validates, binds, chunks, and hands back a corpus view.

RAG-Stack host callers construct ``rag_stack.dataset_manager.DatasetManager``;
that host-facing class specializes this evaluator manager and owns the full
RAG-Stack configuration boundary. Standalone evaluator callers construct
``DatasetEvalManager`` directly with already resolved, pre-chunked data.

Owns:

  * Loading QA + corpus DataFrames (parquet → in-memory + casting + validation).
  * Resume-safe data binding: once a project has a ``data/`` directory, the
    same QA/corpus is locked in for all future runs of that project.
  * Copying datasets into ``<project_dir>/data/`` on first run.
  * Token statistics (avg input/output/chunk tokens) used by the cost model.
    Computed only when a pipeline ``config`` is supplied (owner paths); a
    dataset-only manager (built for a standalone evaluator) leaves it ``None``.
  * The optional ``raw_corpus.parquet`` for chunk-aware sweeps.
  * Per-evaluation corpus artifacts via the hash-keyed chunk cache.
  * Resolving the ``corpus.chunker`` sweep block into a flat runtime dict.

Per-eval contract (both the quality path and the cost-model path use it, so the
chunk is consistent and there is no config side-channel):

  * :meth:`resolve_corpus` — pure, hash-cached; returns a :class:`CorpusView`.
  * :meth:`activate` — writes the chunked corpus to ``data/corpus.parquet``,
    refreshes ``self.corpus_data``, and publishes the frame for same-process
    retrieval nodes (standalone nodes retain a disk fallback).
  * :meth:`apply_corpus_view` — stamps ``_token_stats`` + FAISS ``N`` into a
    pipeline config for the cost model.

This module deliberately does NOT depend on the optimizer / Controller — it only
consumes the YAML config dict and the ``GeneratedDataset`` shape.
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from rag_stack.cost_model.token_stats import TokenStats
from rag_stack.search_space import is_sweepable
# Import the submodule directly (not ``rag_stack.utils``) to avoid the
# ``rag_stack.utils`` <-> static-evaluator circular import — same pattern the
# evaluator itself uses.
from rag_stack.utils.preprocess import (
    cast_corpus_dataset,
    cast_qa_dataset,
    load_data_from_parquet,
    validate_qa_from_corpus_dataset,
    validate_raw_corpus,
)

logger = logging.getLogger("RAG-Stack")

_ACTIVE_CORPUS_FRAMES: weakref.WeakValueDictionary[str, pd.DataFrame] = (
    weakref.WeakValueDictionary()
)
_ACTIVE_CORPUS_LOCK = threading.RLock()


def _project_registry_key(project_dir: str) -> str:
    return os.path.realpath(os.path.abspath(project_dir))


def register_active_corpus(project_dir: str, frame: pd.DataFrame) -> None:
    """Publish the active corpus frame for retrieval nodes in this process."""
    with _ACTIVE_CORPUS_LOCK:
        _ACTIVE_CORPUS_FRAMES[_project_registry_key(project_dir)] = frame


def get_active_corpus(project_dir: str) -> Optional[pd.DataFrame]:
    """Return a strong reference to this process's active project corpus."""
    with _ACTIVE_CORPUS_LOCK:
        return _ACTIVE_CORPUS_FRAMES.get(_project_registry_key(project_dir))


@dataclass
class CorpusView:
    """The chunked-corpus artifacts for a single evaluation.

    Returned by :meth:`DatasetEvalManager.resolve_corpus`. ``token_stats`` is
    ``None`` only on a dataset-only manager (no pipeline config to derive
    them from); ``n_vectors`` is the corpus row count (= number of FAISS
    vectors) for this chunking.
    """

    chunk_hash: str
    corpus_path: str
    token_stats: Optional[TokenStats]
    n_vectors: int


class DatasetEvalManager:
    """Own evaluator QA/corpus frames, artifacts, and per-eval chunking.

    Construction performs the resume-safe data binding: if the project
    already has a ``data/`` directory, supplied paths are ignored (with a
    warning) to keep evaluations from prior runs comparable to new ones.

    After construction, callers read:

      * ``self.qa_data`` (pd.DataFrame)
      * ``self.corpus_data`` (pd.DataFrame)
      * ``self.token_stats`` (TokenStats | None)
      * ``self.raw_corpus_path`` (Optional[str])
      * ``self.project_dir`` (str)

    Owner callers (Controller / QualityStore / PerformanceContext) build this
    WITH a ``config`` and mutate it via :meth:`inject_corpus_size` /
    :meth:`inject_token_stats`. The static evaluator builds a dataset-only
    manager via :meth:`from_dataset` (``config`` absent → no token stats,
    no chunk sweep) and resolves the per-eval corpus from the config passed
    to ``evaluate``.
    """

    def __init__(
        self,
        *,
        project_dir: str,
        config: Optional[dict] = None,
        qa_data_path: Optional[str] = None,
        corpus_data_path: Optional[str] = None,
        dataset: Optional[Any] = None,
    ):
        """Bind the project to its QA + corpus data.

        Args:
            project_dir: absolute path to the project root.
            config: parsed YAML config (mutated by :meth:`inject_corpus_size`
                and :meth:`inject_token_stats`; pass the owner's own config so
                downstream readers see the additions). ``None`` for a
                dataset-only manager — token stats are skipped and chunking
                falls back to the no-op path.
            qa_data_path / corpus_data_path: source parquets, used only on
                first run if the project's ``data/`` directory doesn't yet
                exist.
            dataset: optional :class:`GeneratedDataset` to use directly
                (takes precedence over file paths). Used by callers that
                generated QA in memory.
        """
        self._config = config or {}
        self.project_dir = project_dir
        # Explicit, mandatory dataset name (config ``dataset.dataset_name``) — the
        # human-readable id the IVF cell-imbalance profile is keyed by. Shared via
        # this manager so the controller (load) and eval backend (save) agree.
        # Never derived from a file path. Empty only on a config-less manager.
        self.dataset_name: str = str(
            ((self._config.get("dataset") or {}).get("dataset_name")) or "")
        # Set when a single RAW corpus is passed (no separate corpus.raw_path)
        # and chunk_size is in the search space — that raw corpus is the per-eval
        # chunk source (see _chunk_raw_at_init / _resolve_raw_corpus_path).
        self._single_raw_path: Optional[str] = None
        # A fresh raw-corpus project resolves its default chunk through THE
        # canonical cache during construction. Keep that artifact so eval0 can
        # activate the already-loaded frame without copying/deserializing the
        # same multi-GB parquet again.
        self._initial_chunk_artifacts: Optional[Dict[str, Any]] = None

        in_project_qa = os.path.join(project_dir, "data", "qa.parquet")
        in_project_corpus = os.path.join(project_dir, "data", "corpus.parquet")
        in_project_original_corpus = os.path.join(
            project_dir, "data", "original_corpus.parquet",
        )
        has_in_project_data = (
            os.path.isfile(in_project_qa) and os.path.isfile(in_project_corpus)
        )

        # Per-eval chunker overwrites corpus.parquet with the chunked output
        # (see activate() below). On the next startup, qa.parquet's
        # retrieval_gt UUIDs no longer match the chunker-mangled corpus and
        # validate_qa_from_corpus_dataset rejects the project. Restore the
        # canonical (un-chunked) corpus from original_corpus.parquet if it
        # exists so validation always sees the original UUID set.
        if has_in_project_data and os.path.isfile(in_project_original_corpus):
            import shutil as _shutil
            _shutil.copyfile(in_project_original_corpus, in_project_corpus)
            logger.info(
                "Restored canonical corpus.parquet from original_corpus.parquet "
                "(reverts any per-eval chunker overwrite from a prior run)"
            )

        if dataset is not None:
            self.qa_data = cast_qa_dataset(dataset.qa.data.copy())
            self.corpus_data = cast_corpus_dataset(dataset.corpus.data.copy())
            validate_qa_from_corpus_dataset(self.qa_data, self.corpus_data)
        elif has_in_project_data:
            if (qa_data_path and os.path.abspath(qa_data_path) != in_project_qa) or (
                corpus_data_path and os.path.abspath(corpus_data_path) != in_project_corpus
            ):
                logger.warning(
                    "Ignoring supplied dataset paths — project already has "
                    "data/ from the original run; using those for resume "
                    f"safety (qa={in_project_qa}, corpus={in_project_corpus})."
                )
            self.qa_data, self.corpus_data = load_data_from_parquet(
                in_project_qa, in_project_corpus,
            )
        else:
            if not qa_data_path or not corpus_data_path:
                raise ValueError(
                    f"Project '{project_dir}' has no data/ directory and "
                    f"no qa_data_path/corpus_data_path was supplied."
                )
            corpus_for_init = corpus_data_path
            # Single raw-corpus input: when the chunker is a search-space
            # dimension and no separate corpus.raw_path is configured, the
            # corpus you pass IS the RAW corpus. Validate it's raw, then chunk
            # it ONCE (default chunk_size) to produce a valid chunked corpus for
            # init; the raw stays the per-eval sweep source.
            cfg_raw = ((self._config.get("algo_search_space") or {}).get("corpus") or {}).get("raw_path")
            # rag-stack ALWAYS ingests a RAW corpus and chunks it — a FIXED chunk_size
            # STILL needs chunking; chunking is NOT gated on chunk being a swept dim.
            # Chunk the input here whenever it is a raw corpus (doc_id+contents, not
            # already pre-chunked). A separate corpus.raw_path means the passed corpus
            # is already the chunked one, so leave it. (resolve_corpus_chunker + the
            # per-eval resolve_corpus carry the fixed chunk value, so this is consistent.)
            if not cfg_raw and self._is_raw_corpus(corpus_data_path):
                self._single_raw_path = os.path.abspath(corpus_data_path)
                corpus_for_init = self._chunk_raw_at_init(
                    corpus_data_path, qa_data_path,
                )
            self.qa_data, self.corpus_data = load_data_from_parquet(
                qa_data_path, corpus_for_init,
            )

        # Usually set only by activate(). A fresh raw-corpus project is the one
        # exception: its startup corpus now comes directly from a complete,
        # immutable chunk-cache entry, so it is safe to remember that view and
        # let eval0 reuse the in-memory frame.
        self._active_view_key: Optional[tuple[str, str, int, int]] = None
        initial_corpus_path = (
            str(self._initial_chunk_artifacts["corpus_path"])
            if self._initial_chunk_artifacts is not None
            else None
        )
        self._copy_data_to_project(corpus_source_path=initial_corpus_path)
        if self._initial_chunk_artifacts is not None:
            source_path = os.path.realpath(os.path.abspath(initial_corpus_path))
            source_stat = os.stat(source_path)
            self._active_view_key = (
                str(self._initial_chunk_artifacts["chunk_hash"]),
                source_path,
                source_stat.st_size,
                source_stat.st_mtime_ns,
            )
            register_active_corpus(self.project_dir, self.corpus_data)

        # Raw corpus for chunk-aware sweeps.
        self.raw_corpus_path: Optional[str] = self._resolve_raw_corpus_path()

        # Constraint: when the chunker is a search-space dimension, the corpus is
        # re-chunked per eval, so a RAW (un-chunked) corpus is REQUIRED and must
        # be genuinely raw. Otherwise chunk sweeps silently no-op (or re-chunk
        # already-chunked passages). Fail loudly at startup with a clear message.
        if self._chunk_in_search_space():
            if self.raw_corpus_path is None:
                raise ValueError(
                    "chunk_size/chunker is in the search space but no RAW corpus "
                    "is available. Provide it via 'corpus.raw_path' in the YAML "
                    "(or pass a raw --corpus-data). Chunk sweeps need the raw "
                    "corpus to re-chunk per config."
                )
            validate_raw_corpus(pd.read_parquet(self.raw_corpus_path), require_raw=True)

        # Token statistics (consumed by the cost model and exposed back via
        # ``config["_token_stats"]`` in :meth:`inject_token_stats`). Only owner
        # paths supply a config; a dataset-only manager leaves these unset.
        self.token_stats: Optional[TokenStats] = None
        if self._config:
            initial_stats = (
                self._initial_chunk_artifacts.get("token_stats")
                if self._initial_chunk_artifacts is not None
                else None
            )
            # A cache MISS built these stats from the canonical chunk and kept
            # its tokenizer attached, so reuse them. A pre-existing cache hit
            # reconstructs stats from JSON (no tokenizer); rebuild the live
            # object once, as startup already did before this change.
            if initial_stats is not None and getattr(
                initial_stats, "_tokenizer", None,
            ) is not None:
                self.token_stats = initial_stats
            else:
                self.token_stats = TokenStats(
                    self.qa_data, self.corpus_data, self._config,
                )

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        project_dir: str,
        config: Optional[dict] = None,
    ) -> "DatasetEvalManager":
        """Build a manager directly from an in-memory ``GeneratedDataset``.

        Used by the static evaluator when no owner-supplied manager is passed
        (standalone / tests). Without a ``config`` the manager owns the corpus
        and the chunk cache but skips token stats and chunk sweeps.
        """
        return cls(project_dir=project_dir, config=config, dataset=dataset)

    def _chunk_in_search_space(self) -> bool:
        """True when the corpus chunker (chunk_size / component / overlap) is a
        search-space dimension — i.e. the corpus is re-chunked PER EVAL with a
        DIFFERENT chunk each time. (A FIXED chunk is still chunked — see
        _is_raw_corpus / the init gate — just identically every eval.)"""
        chunker = ((self._config.get("algo_search_space") or {}).get("corpus") or {}).get("chunker") or {}
        return any(
            is_sweepable(chunker.get(k))
            for k in ("chunk_size", "component", "chunk_overlap")
        )

    @staticmethod
    def _is_raw_corpus(path: str) -> bool:
        """True if ``path`` is a RAW (un-chunked) corpus that rag-stack should chunk:
        it carries {doc_id, contents} (or legacy ``texts``) and is NOT already
        pre-chunked (``start_end_idx`` absent). Pre-chunked or non-corpus inputs
        return False → used as-is. Reads only the parquet SCHEMA (cheap)."""
        try:
            import pyarrow.parquet as _pq
            cols = set(_pq.read_schema(path).names)
        except Exception:
            try:
                cols = set(pd.read_parquet(path).columns)
            except Exception:
                return False
        has_text = ("contents" in cols) or ("texts" in cols)
        return ("doc_id" in cols) and has_text and ("start_end_idx" not in cols)

    # ------------------------------------------------------------------
    # Construction helpers (private — called from __init__).
    # ------------------------------------------------------------------

    def _copy_data_to_project(
        self, *, corpus_source_path: Optional[str] = None,
    ) -> None:
        """Copy datasets to ``<project_dir>/data/`` if not already there.

        Also snapshots a canonical ``original_corpus.parquet`` so that
        per-eval chunker writes to ``corpus.parquet`` are reversible — on
        the next startup we restore from the snapshot before validation.
        """
        data_dir = os.path.join(self.project_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        qa_path = os.path.join(data_dir, "qa.parquet")
        if not os.path.exists(qa_path):
            self.qa_data.to_parquet(qa_path, index=False)
        corpus_path = os.path.join(data_dir, "corpus.parquet")
        if not os.path.exists(corpus_path):
            if corpus_source_path is not None:
                import shutil as _shutil
                _shutil.copyfile(corpus_source_path, corpus_path)
            else:
                self.corpus_data.to_parquet(corpus_path, index=False)
        original_corpus_path = os.path.join(data_dir, "original_corpus.parquet")
        if not os.path.exists(original_corpus_path):
            import shutil as _shutil
            _shutil.copyfile(corpus_path, original_corpus_path)

    def _chunk_raw_at_init(self, raw_path: str, qa_path: str) -> str:
        """Resolve the default raw-corpus view through the canonical cache.

        ``TokenStats`` normally exists before :func:`chunk_cache.get_or_build`
        is called. Fresh projects are the bootstrap exception: they need a
        chunked corpus in order to construct their first stats. Supply a
        factory so the cache build creates both artifacts in one pass. Eval0
        then hits this exact entry instead of repeating the chunk operation.
        """
        from rag_stack_evaluator.static_rag_evaluator.chunk_cache import get_or_build

        chunker = ((self._config.get("algo_search_space") or {}).get("corpus") or {}).get("chunker") or {}

        def _first(v, default):
            if isinstance(v, list):
                return v[0] if v else default
            return v if v is not None else default

        method = str(_first(chunker.get("component"), "recursivecharacter"))
        size = int(_first(chunker.get("chunk_size"), 512))
        overlap = int(_first(chunker.get("chunk_overlap"), 0))
        # A file-backed, config-less DatasetEvalManager deliberately owns no token
        # statistics. Preserve that uncommon standalone bootstrap path; the
        # optimizer/quality owner path always supplies a config and therefore
        # uses the canonical cache below.
        if not self._config:
            from rag_stack_evaluator.static_rag_evaluator.chunk_cache import _run_chunker

            raw_df = pd.read_parquet(raw_path)
            if "contents" not in raw_df.columns and "texts" in raw_df.columns:
                raw_df = raw_df.rename(columns={"texts": "contents"})
            chunked = _run_chunker(raw_df, method, size, overlap)
            out_dir = os.path.join(self.project_dir, "data")
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, "_init_chunked_corpus.parquet")
            chunked.to_parquet(out, index=False)
            return out

        qa_data = cast_qa_dataset(pd.read_parquet(qa_path, engine="pyarrow"))
        artifacts = get_or_build(
            project_dir=self.project_dir,
            raw_corpus_path=raw_path,
            fallback_corpus_path=raw_path,
            chunk_params={
                "component": method,
                "chunk_size": size,
                "chunk_overlap": overlap,
            },
            token_stats_factory=lambda chunked: TokenStats(
                qa_data, chunked, self._config,
            ),
        )
        self._initial_chunk_artifacts = artifacts
        logger.info(
            "Initialized default corpus from canonical chunk cache: "
            f"chunk_hash={artifacts['chunk_hash']} "
            f"(size={size}, method={method}, overlap={overlap})"
        )
        return str(artifacts["corpus_path"])

    def _resolve_raw_corpus_path(self) -> Optional[str]:
        """Decide where the raw (un-chunked) corpus parquet lives.

        Resume-safe: an existing ``<project>/data/raw_corpus.parquet`` wins.
        Otherwise copy in from ``corpus.raw_path`` in the YAML, or from the
        single raw corpus passed as the corpus input. If none is present,
        return ``None`` and chunk sweeps become no-ops.
        """
        in_project_raw = os.path.join(self.project_dir, "data", "raw_corpus.parquet")
        corpus_block = (self._config.get("algo_search_space") or {}).get("corpus") or {}
        cfg_raw_path = corpus_block.get("raw_path") if isinstance(corpus_block, dict) else None

        if os.path.isfile(in_project_raw):
            return in_project_raw
        if cfg_raw_path:
            cfg_raw_path = os.path.expandvars(str(cfg_raw_path))
            if os.path.isfile(cfg_raw_path):
                import shutil as _shutil
                os.makedirs(os.path.dirname(in_project_raw), exist_ok=True)
                _shutil.copy2(cfg_raw_path, in_project_raw)
                logger.info(f"Copied raw corpus into project: {in_project_raw}")
                return in_project_raw
            logger.warning(
                f"corpus.raw_path '{cfg_raw_path}' not found — chunk "
                f"sweeps will be no-ops; the existing corpus.parquet "
                f"will be used for every eval."
            )
        # Single raw-corpus input: the raw corpus was passed AS the corpus input
        # (no separate corpus.raw_path). Persist it as the canonical raw source
        # so resume + per-eval chunking find it.
        if self._single_raw_path and os.path.isfile(self._single_raw_path):
            import shutil as _shutil
            os.makedirs(os.path.dirname(in_project_raw), exist_ok=True)
            _shutil.copy2(self._single_raw_path, in_project_raw)
            logger.info(f"Single raw-corpus input -> {in_project_raw}")
            return in_project_raw
        return None

    # ------------------------------------------------------------------
    # YAML-config mutators — called once during owner __init__.
    # ------------------------------------------------------------------

    def inject_corpus_size(self) -> None:
        """Auto-inject ``N`` (corpus size) into FAISS vectordb configs.

        Only ``faiss_ivf`` / ``faiss_hnsw`` need ``N``; other vectordb
        types are left alone. An explicit ``N`` in the YAML wins.
        """
        corpus_size = len(self.corpus_data)
        from rag_stack.search_space import algo_vectordb_blocks

        for vdb in algo_vectordb_blocks(self._config):
            if vdb.get("db_type") not in ("faiss_ivf", "faiss_hnsw"):
                continue
            if "N" not in vdb:
                vdb["N"] = corpus_size
                logger.info(
                    f"Auto-inferred N={corpus_size} for vectordb '{vdb.get('name')}'"
                )

    def inject_token_stats(self) -> None:
        """Stash token statistics into the YAML config.

        Two consumers read these later:
          * StaticRAGEvaluatorQualityOnly / StaticAssembly: ``config["_token_stats"]``.
          * RAGO's hardware sweep: ``config["system"]["cm_search_space"]``.
        """
        if self.token_stats is None:
            raise RuntimeError(
                "inject_token_stats requires a config-backed manager "
                "(token_stats was not computed)."
            )
        self._config["_token_stats"] = self.token_stats.to_dict()
        system = self._config.get("system", {})
        search_space = system.get("cm_search_space", {})
        search_space["dec_steps"] = self.token_stats.avg_output_tokens
        search_space["seq_len_inference_prefill"] = (
            self.token_stats.prompt_template_tokens
            + 10 * self.token_stats.avg_chunk_tokens
        )

    # ------------------------------------------------------------------
    # QA mutation (rare — only used when an external caller swaps QA).
    # ------------------------------------------------------------------

    def update_qa_data(self, new_qa_data: pd.DataFrame, evaluator=None) -> None:
        """Replace the QA dataframe and propagate to the evaluator + disk."""
        self.qa_data = new_qa_data
        if evaluator is not None:
            evaluator.qa_data = new_qa_data
        data_dir = os.path.join(self.project_dir, "data")
        self.qa_data.to_parquet(os.path.join(data_dir, "qa.parquet"), index=False)

    # ------------------------------------------------------------------
    # Per-eval resolution.
    # ------------------------------------------------------------------

    def resolve_corpus_chunker(self, decoded_config: dict) -> Dict[str, Any]:
        """Build the ``corpus_runtime.chunker`` dict for one evaluation.

        Combines YAML defaults (scalar / single-element) with the decoded
        optimizer overrides under ``decoded_config["corpus"]["chunker"]``.
        Returns ``{}`` when neither source produced any chunker keys, which
        the caller should interpret as "no chunker block needed."
        """
        corpus_overrides = (decoded_config.get("corpus") or {}).get("chunker") or {}
        base_corpus = (self._config.get("algo_search_space") or {}).get("corpus") or {}
        base_chunker = base_corpus.get("chunker") if isinstance(base_corpus, dict) else None
        chunker_runtime: Dict[str, Any] = {}

        # Carry over scalar YAML defaults that weren't swept.
        if isinstance(base_chunker, dict):
            for k, v in base_chunker.items():
                if not is_sweepable(v):
                    chunker_runtime[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

        # Apply optimizer-decoded overrides on top.
        for k, v in corpus_overrides.items():
            chunker_runtime[k] = v
        return chunker_runtime

    def resolve_corpus(self, chunker_params: Optional[Dict[str, Any]]) -> CorpusView:
        """Build / load the chunked-corpus cache entry for one evaluation.

        Pure and hash-cached: identical ``chunker_params`` (+ unchanged raw
        corpus) always return the same ``chunk_hash`` and on-disk artifacts,
        so the quality path and the cost-model path stay consistent without a
        config side-channel. Call :meth:`activate` to make the result the
        active ``data/corpus.parquet`` that retrieval nodes read.
        """
        from rag_stack_evaluator.static_rag_evaluator.chunk_cache import get_or_build

        in_project_corpus = os.path.join(self.project_dir, "data", "corpus.parquet")
        artifacts = get_or_build(
            project_dir=self.project_dir,
            raw_corpus_path=self.raw_corpus_path,
            fallback_corpus_path=in_project_corpus,
            chunk_params=chunker_params or {},
            base_token_stats=self.token_stats,
        )
        corpus_path = artifacts["corpus_path"]
        return CorpusView(
            chunk_hash=artifacts["chunk_hash"],
            corpus_path=corpus_path,
            token_stats=artifacts["token_stats"],
            n_vectors=self._num_rows(corpus_path),
        )

    def activate(self, view: CorpusView) -> pd.DataFrame:
        """Make ``view`` the active corpus on disk + in memory for this eval.

        A non-no-op chunking is copied to ``<project>/data/corpus.parquet`` for
        standalone/spawned node compatibility. The active frame is also
        published process-locally so evaluator-owned retrieval nodes can reuse
        it without another parquet read. Repeated activation of the same
        immutable cache view is a no-op. Returns the active ``corpus_data``.
        """
        from rag_stack_evaluator.static_rag_evaluator.chunk_cache import NO_OP_HASH

        in_project_corpus = os.path.join(self.project_dir, "data", "corpus.parquet")
        source_path = os.path.realpath(os.path.abspath(view.corpus_path))
        source_stat = os.stat(source_path)
        view_key = (
            view.chunk_hash,
            source_path,
            source_stat.st_size,
            source_stat.st_mtime_ns,
        )

        # A Controller evaluates candidates sequentially. Repeated activation
        # of the same immutable cache view must not copy and deserialize the
        # multi-gigabyte corpus again on every evaluation.
        if self._active_view_key == view_key and os.path.isfile(in_project_corpus):
            register_active_corpus(self.project_dir, self.corpus_data)
            return self.corpus_data

        if (
            view.chunk_hash != NO_OP_HASH
            and source_path != os.path.realpath(os.path.abspath(in_project_corpus))
        ):
            import shutil as _shutil
            os.makedirs(os.path.dirname(in_project_corpus), exist_ok=True)
            _shutil.copyfile(source_path, in_project_corpus)
            self.corpus_data = pd.read_parquet(in_project_corpus)

        self._active_view_key = view_key
        register_active_corpus(self.project_dir, self.corpus_data)
        return self.corpus_data

    def apply_corpus_view(self, pipeline_config: dict, view: CorpusView) -> None:
        """Stamp a corpus view's cost-model inputs into ``pipeline_config``.

        Sets ``_token_stats`` (avg_chunk_tokens for this chunking) and the
        per-config FAISS ``N`` (= number of CHUNKS, which varies with
        chunk_size: smaller chunks → more vectors → different IVF-PQ / HNSW
        latency). Consumed by StaticAssembly on the cost-model path. Also resolves
        any relative IVF-PQ ``nlist_factor`` into a concrete ``nlist`` now that
        the true chunk count is known (see :meth:`resolve_nlist_factor`).
        """
        if view.token_stats is not None:
            pipeline_config["_token_stats"] = view.token_stats.to_dict()
        for vdb in pipeline_config.get("vectordb", []):
            if vdb.get("db_type") in ("faiss_ivf", "faiss_hnsw"):
                vdb["N"] = view.n_vectors
        self.resolve_nlist_factor(pipeline_config.get("vectordb", []), view.n_vectors)

    @staticmethod
    def resolve_nlist_factor(vectordb_configs, n_vectors: int) -> None:
        """Resolve a relative IVF-PQ ``nlist_factor`` into a concrete ``nlist``.

        ``nlist`` (the IVF Voronoi-cell / k-means centroid count) must not
        exceed the number of training vectors ``N``; ``nlist > N`` makes FAISS
        k-means untrainable and the arm fails as INVALID. ``N`` here is the
        per-eval CHUNK count, which swings with the chunker (~344 chunks at
        character@2048 vs ~10k at recursivecharacter@128), so a single absolute
        ``nlist`` list cannot stay valid across the chunker sweep. Expressing
        the search dimension as a factor of ``sqrt(N)`` ties the cluster count
        to the actual corpus size::

            nlist = clamp(round(factor * sqrt(N)), 1, N)

        the textbook FAISS heuristic (nlist in the sqrt(N)..16*sqrt(N) band),
        which guarantees ``nlist <= N`` by construction. Mutates each
        ``faiss_ivf`` vdb in place — pops ``nlist_factor`` and sets ``nlist``.
        Absolute ``nlist`` configs (no factor) are left untouched, so this is
        backward compatible. No-op for non-IVF stores (HNSW has no nlist).
        """
        n = max(1, int(n_vectors))
        for vdb in vectordb_configs or []:
            if vdb.get("db_type") != "faiss_ivf" or "nlist_factor" not in vdb:
                continue
            factor = float(vdb.pop("nlist_factor"))
            nlist = round(factor * (n ** 0.5))
            vdb["nlist"] = max(1, min(int(nlist), n))

    @staticmethod
    def _num_rows(parquet_path: str) -> int:
        """Row count of a parquet without loading it (reads footer metadata)."""
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
