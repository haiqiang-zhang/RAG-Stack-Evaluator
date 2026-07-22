"""Import-order contract for the split host and evaluator packages."""

import subprocess
import sys


def test_host_search_space_can_import_before_evaluator_dataset_owner():
	code = """
from rag_stack.search_space import RagSearchSpace
from rag_stack_evaluator.static_rag_evaluator import DatasetEvalManager
assert RagSearchSpace.__name__ == 'RagSearchSpace'
assert DatasetEvalManager.__name__ == 'DatasetEvalManager'
"""
	completed = subprocess.run(
		[sys.executable, "-c", code],
		check=False,
		capture_output=True,
		text=True,
	)
	assert completed.returncode == 0, completed.stderr
