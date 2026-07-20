import gc
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class BaseModule(metaclass=ABCMeta):
	@abstractmethod
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		pass

	@abstractmethod
	def _pure(self, *args, **kwargs):
		pass

	@classmethod
	def run_evaluator(
		cls,
		project_dir: Union[str, Path],
		previous_result: pd.DataFrame,
		*args,
		**kwargs,
	):
		"""Run the module on `previous_result` and return its output.

		If a `cache` kwarg is supplied (a `ModelCache` instance), the cache
		owns model lifetimes — we skip the per-call `del instance + gc +
		torch.cuda.empty_cache()` cleanup that inflated measured latency
		4-50x in the legacy quality-only path. Modules consult the cache
		inside `__init__` / `pure` to avoid re-loading weights.

		If `cache` is None (legacy path), the original fresh-per-call
		lifecycle is preserved — instance is dropped and CUDA cache cleared
		after the call so quality-only callers keep working unchanged.
		"""
		cache = kwargs.get("cache", None)
		instance = cls(project_dir, *args, **kwargs)
		if cache is not None:
			# Cache-managed lifetimes — no per-call teardown.
			return instance.pure(previous_result.copy(deep=True), *args, **kwargs)
		# Legacy path: fresh-per-call lifecycle.
		try:
			return instance.pure(previous_result.copy(deep=True), *args, **kwargs)
		finally:
			del instance
			gc.collect()
			try:
				import torch
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			except ImportError:
				pass

	@abstractmethod
	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		"""
		This function is for cast function (a.k.a decorator) only for pure function in the whole node.
		"""
		pass
