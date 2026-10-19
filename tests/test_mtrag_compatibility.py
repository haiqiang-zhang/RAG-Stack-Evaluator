"""The built-in backend's optional MT-RAG imports remain lazy aliases."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


PROVIDER_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "legacy_mtrag_test", PROVIDER_ROOT / "rag_stack_evaluator/integrations/mtrag.py",
)
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def test_legacy_import_forwards_to_optional_host_example():
    scoring = SimpleNamespace(score_with_mtrag=object())
    with patch.object(legacy, "import_module", return_value=scoring) as load:
        assert legacy.score_with_mtrag is scoring.score_with_mtrag
        load.assert_called_once_with("rag_stack_mtrag.scoring")


def test_missing_example_reports_host_install_location():
    absent = ModuleNotFoundError("Optional package absent", name="rag_stack_mtrag")
    with patch.object(legacy, "import_module", side_effect=absent):
        try:
            legacy.mt_rag_faithfulness
        except ImportError as error:
            assert "python -m pip install --no-deps -e examples/mtrag" in str(error)
        else:
            raise AssertionError("Missing optional plugin must produce an installation hint")


def test_legacy_module_import_does_not_load_plugin_or_model_libraries():
    code = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
blocked = {'rag_stack_mtrag', 'torch', 'transformers', 'ragas', 'pandas', 'vllm'}

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Compatibility import loaded optional package: ' + fullname)

sys.meta_path.insert(0, BlockOptionalImports())
import rag_stack_evaluator.integrations.mtrag
assert not blocked.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(PROVIDER_ROOT)], check=True)
