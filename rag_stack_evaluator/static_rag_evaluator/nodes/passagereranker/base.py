# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

import abc
import logging
import sys
from pathlib import Path
from typing import Union

import pandas as pd

from rag_stack_evaluator.static_rag_evaluator.schema import BaseModule
from rag_stack.utils.preprocess import validate_qa_dataset
from rag_stack_evaluator.static_rag_evaluator.utils.cast import cast_retrieve_infos

logger = logging.getLogger("RAG-Stack")


class BasePassageReranker(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(
			f"Initialize passage reranker node - {self.__class__.__name__} module..."
		)

	def _load_replicas(self, cache, component, model_name, device, devices, factory):
		"""Populate ``self._replicas`` (one model per device, data-parallel) and
		set ``self.device`` / ``self.model`` (the first replica, for back-compat).

		``devices`` is the injected per-engine GPU list (measured multi-GPU);
		absent/empty → a single replica on ``device``. With a ``cache`` each
		replica is pulled from :meth:`ModelCache.get_reranker_replicas` (cached
		per device, so a warm single device short-circuits). Returns the replica
		list. See :mod:`.dp` for how the replicas are used (data parallelism)."""
		devs = [str(d) for d in (devices or [device]) if d]
		if not devs:
			devs = [str(device)]
		self._replica_devices = devs
		if cache is not None:
			self._cache_owned = True
			self._replicas = cache.get_reranker_replicas(
				component, model_name, devs, factory,
			)
		else:
			self._cache_owned = False
			# Measured aux stages live in a spawned child, where the parent-side
			# ModelCache is intentionally unavailable.  ``cache is None`` therefore
			# does not imply a quality-only single-device run: honor every injected
			# device and build one independent replica per card.
			self._replicas = [
				factory(component, model_name, replica_device)
				for replica_device in devs
			]
		self.device = devs[0]
		self.model = self._replicas[0]
		return self._replicas

	def __del__(self):
		if sys.is_finalizing():
			return
		logger.info(
			f"Deleting passage reranker node - {self.__class__.__name__} module..."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		# Measured serving calls this once per dynamic batch.  INFO here makes
		# synchronous console/file rendering part of the service time.
		logger.debug(
			"Running passage reranker node - %s module...",
			self.__class__.__name__,
		)
		validate_qa_dataset(previous_result)

		# find queries columns
		assert "query" in previous_result.columns, (
			"previous_result must have query column."
		)
		queries = previous_result["query"].tolist()

		retrieve_infos = cast_retrieve_infos(previous_result)
		return (
			queries,
			retrieve_infos["retrieved_contents"],
			retrieve_infos["retrieve_scores"],
			retrieve_infos["retrieved_ids"],
		)
