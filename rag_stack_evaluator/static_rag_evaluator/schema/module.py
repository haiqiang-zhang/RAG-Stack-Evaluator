# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict

from rag_stack_evaluator.static_rag_evaluator.support import get_support_modules


@dataclass
class Module:
	component: str
	module_param: Dict
	module: Callable = field(init=False)

	def __post_init__(self):
		self.module = get_support_modules(self.component)
		if self.module is None:
			raise ValueError(f"Module type {self.component} is not supported.")

	@classmethod
	def from_dict(cls, module_dict: Dict) -> "Module":
		_module_dict = deepcopy(module_dict)
		component = _module_dict.pop("component")
		module_params = _module_dict
		return cls(component, module_params)
