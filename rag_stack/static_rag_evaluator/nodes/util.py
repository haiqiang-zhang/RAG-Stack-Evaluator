# Portions derived from AutoRAG (https://github.com/Marker-Inc-Korea/AutoRAG), Apache-2.0.
# Modified by the RAG-Stack authors for namespace and runtime integration; see LICENSE.autorag and NOTICE.

from typing import Optional, Dict

from rag_stack.static_rag_evaluator.nodes.generator.registry import get_generator_class


def make_generator_callable_param(generator_dict: Optional[Dict]):
	if "generator_backend" not in generator_dict.keys():
		generator_dict = {
			"generator_backend": "openai_llm",
			"model": "gpt-4o-mini",
		}
	module_str = generator_dict.pop("generator_backend")
	module_class = get_generator_class(module_str)
	module_param = generator_dict
	return module_class, module_param
