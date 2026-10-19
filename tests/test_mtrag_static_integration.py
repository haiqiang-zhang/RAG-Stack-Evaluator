"""CPU integration at the real static scoring boundary; pipeline output is a fixture.

Run with the host CPU development environment. No pipeline/model is executed.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

# A host checkout supplies the example source; standalone provider checkouts
# use the independently installed rag-stack-mtrag-example package instead.
host_example = Path(__file__).resolve().parents[2] / "examples/mtrag"
if (host_example / "rag_stack_mtrag").is_dir():
    sys.path.insert(0, str(host_example))

from rag_stack_evaluator.integrations.mtrag import to_qa_rows


def test_registered_metric_dispatch_objective_override_and_trace(tmp_path):
    from rag_stack_evaluator.factory import supported_metrics
    from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
    from rag_stack.utils.objectives import aggregate_quality_metrics
    from rag_stack.utils.preprocess import cast_qa_dataset

    tasks = [{
        "task_id": "conversation<::>2",
        "input": [{"speaker": "user", "text": "A question with an emoji 🙂  "}],
        "targets": [{"text": "Reference answer, not the prediction"}],
        "contexts": [{"text": "Reference context, not retrieved evidence"}],
    }]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(tasks[0]) + "\n")
    scorer = tmp_path / "contract_scorer.py"
    scorer.write_text(
        "import json, pathlib, sys\n"
        "rows = [json.loads(s) for s in pathlib.Path(sys.argv[1]).read_text().splitlines()]\n"
        "for row in rows:\n"
        "    assert row['predictions'] == [{'text': 'Actual fixture answer'}]\n"
        "    assert row['contexts'] == [{'text': 'Actual final compressed context'}]\n"
        "    row['metrics'] = {'RL_F': [0.625]}  # synthetic transport value\n"
        "pathlib.Path(sys.argv[2]).write_text(''.join(json.dumps(r)+'\\n' for r in rows))\n"
    )
    metric = {
        "metric_name": "mt_rag_faithfulness",
        "tasks_path": str(tasks_path),
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "scorer_command": [sys.executable, "-S", str(scorer), "{input}", "{output}"],
        "scorer_identity": "synthetic-static-integration-fixture",
        "upstream_revision": "53fef6f5917c7e10c436f697e69a955dbc795d81",
    }
    evaluator = object.__new__(StaticRAGEvaluatorQualityOnly)
    evaluator.qa_data = cast_qa_dataset(pd.DataFrame(to_qa_rows(tasks)))
    final = pd.DataFrame({
        "generated_texts": ["Actual fixture answer"],
        "retrieved_contents": [["Actual final compressed context"]],
        "retrieved_contents_semantic": [["Original uncompressed context"]],
    })
    assert "mt_rag_faithfulness" in supported_metrics("static_gt")
    assert "mt_rag_faithfulness" in evaluator._ALL_METRIC_FUNC_DICT
    # Test the real public evaluate/override method and metric dispatch, while
    # replacing only pipeline execution with a fixture to avoid models/GPU use.
    trace_fixture = {"interface_test_only": True}
    callbacks = []

    def fixture_pipeline(config, run_dir=None, on_trace_ready=None):
        on_trace_ready(trace_fixture)
        quality = evaluator._evaluate_final_result(final, config["eval_backend_setting"])
        return SimpleNamespace(quality=quality, trace=trace_fixture)

    config = {"eval_backend_setting": {"metrics": []}}
    original = deepcopy(config)
    with patch.object(evaluator, "_run_pipeline", side_effect=fixture_pipeline):
        result = evaluator.evaluate(
            config, metrics_override=[metric], on_trace_ready=callbacks.append,
        )
    assert config == original
    assert result.pop("__execution_dag__") is callbacks[0] is trace_fixture
    assert result == {"mt_rag_faithfulness": 0.625}
    assert aggregate_quality_metrics(
        result, {"mode": "mean", "metrics": ["mt_rag_faithfulness"]}
    ) == 0.625
