"""CPU contract tests; the scoring subprocess emits explicitly synthetic scores."""

from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/mtrag"
sys.path.insert(0, str(EXAMPLE))

from rag_stack_evaluator.api import Evaluator, normalize_evaluation_result
from rag_stack_evaluator import factory
from rag_stack_evaluator.integrations.mtrag import score_with_mtrag
from rag_stack_mtrag.backend import BACKEND, MTRAGEvaluator, PipelineRun, Prediction
from rag_stack_mtrag.fixture import fixture_config, fixture_tasks, run_pipeline
from rag_stack_mtrag import run_official


class Qids:
    def __init__(self, ids):
        self.ids = ids

    def tolist(self):
        return self.ids


class MTRAGBackendExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mtrag_backend_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tasks = fixture_tasks()
        self.task_path = self.root / "tasks.jsonl"
        self.task_path.write_text("".join(json.dumps(task) + "\n" for task in self.tasks))
        self.options = {
            "tasks_path": str(self.task_path),
            "tasks_sha256": hashlib.sha256(self.task_path.read_bytes()).hexdigest(),
            "pipeline": "rag_stack_mtrag.fixture:run_pipeline",
            "pipeline_identity": "synthetic-pipeline-v1",
            "scorer_command": [sys.executable, str(EXAMPLE / "rag_stack_mtrag/fixture.py"),
                               "--input", "{input}", "--output", "{output}"],
            "scorer_identity": "synthetic-scores-v1",
            "upstream_revision": "53fef6f5917c7e10c436f697e69a955dbc795d81",
        }

    def evaluator(self, pipeline=None, manager=None):
        return MTRAGEvaluator(factory.EvaluatorContext(options=self.options, dataset_manager=manager),
                             pipeline=pipeline)

    def test_serving_trace_published_with_workflow_before_full_scoring_transport(self):
        observed = []
        requests_seen = []

        def pipeline(config, requests, recorder, *, options):
            requests_seen.extend(deepcopy(requests))
            return run_pipeline(config, requests, recorder, options=options)

        def score(*args, **kwargs):
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["provenance"]["workflow_schema"]["dataflow_type"], "sequential")
            return score_with_mtrag(*args, **kwargs)

        evaluator = self.evaluator(pipeline)
        self.assertIsInstance(evaluator, Evaluator)
        with patch("rag_stack_mtrag.backend.score_with_mtrag", side_effect=score):
            result = evaluator.evaluate(fixture_config(), run_dir=str(self.root / "run"),
                                        on_trace_ready=observed.append)
        self.assertIs(result.trace, observed[0])
        self.assertEqual(result.metrics, {"mt_rag_faithfulness": 0.375,
                                         "mt_rag_radbench": 0.5, "mt_rag_aggregate": 0.1875})
        self.assertTrue(all(set(request) == {"task_id", "input"} for request in requests_seen))
        for query in result.trace["queries"]:
            self.assertEqual([call["stage"] for call in query["calls"]],
                             ["lexical_retrieval", "generator"])
        flat = normalize_evaluation_result(result, require_trace=True)
        self.assertIs(flat["__execution_dag__"], result.trace)
        self.assertEqual(flat["__workflow_schema__"], result.workflow)

    def test_actual_pipeline_outputs_replace_reference_evidence(self):
        self.evaluator().evaluate(fixture_config(), run_dir=str(self.root / "run"))
        audit = next((self.root / "run/mtrag").iterdir())
        records = [json.loads(line) for line in (audit / "input.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["predictions"], [{"text": "The example service supports private deployment."}])
        self.assertEqual(records[0]["contexts"], [{"text": "The example service supports private deployment."}])
        self.assertEqual(records[0]["targets"], self.tasks[0]["targets"])
        self.assertEqual(records[0]["input"], self.tasks[0]["input"])

    def test_candidate_knob_changes_actual_contexts_and_recorded_generator_input(self):
        evaluator = self.evaluator()
        one = evaluator.evaluate(fixture_config())
        config = fixture_config()
        config["node_lines"][0]["nodes"][0]["modules"][0]["top_k"] = 2
        two = evaluator.evaluate(config, run_dir=str(self.root / "two_contexts"))
        self.assertGreater(two.trace["queries"][0]["calls"][-1]["input_tokens"],
                           one.trace["queries"][0]["calls"][-1]["input_tokens"])
        audit = next((self.root / "two_contexts/mtrag").iterdir())
        record = json.loads((audit / "input.jsonl").read_text().splitlines()[0])
        self.assertEqual(len(record["contexts"]), 2)

    def test_unsupported_runtime_is_rejected_before_pipeline_execution(self):
        evaluator = self.evaluator()
        for runtime in ({"mode": "react"}, {"mode": "sequential", "rag_dataflow": "react"}):
            config = fixture_config()
            config["pipeline_runtime"] = runtime
            with self.subTest(runtime=runtime), patch.object(evaluator, "pipeline") as pipeline:
                with self.assertRaisesRegex(ValueError, "sequential"):
                    evaluator.evaluate(config)
                pipeline.assert_not_called()

    def test_active_host_qa_subset_controls_scoring_and_trace(self):
        qid = self.tasks[1]["task_id"]
        manager = SimpleNamespace(qa_data={"qid": Qids([qid])})
        result = self.evaluator(manager=manager).evaluate(fixture_config())
        self.assertEqual([row["question_id"] for row in result.trace["queries"]], [qid])
        self.assertEqual(result.metrics["mt_rag_faithfulness"], 0.25)

    def test_bad_host_qa_ids_are_rejected(self):
        for ids in ([], ["unknown"], [2], [self.tasks[0]["task_id"]] * 2):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.evaluator(manager=SimpleNamespace(qa_data={"qid": Qids(ids)})).evaluate(fixture_config())

    def test_metric_override_selects_results_without_mutating_defaults(self):
        evaluator = self.evaluator()
        result = evaluator.evaluate(fixture_config(), metrics_override=[
            {"metric_name": "mt_rag_aggregate_idk", "description": "test selection"}])
        self.assertEqual(result.metrics, {"mt_rag_aggregate_idk": 0.1875})
        self.assertEqual(len(evaluator.metrics), 3)
        config = fixture_config()
        config["eval_backend_setting"] = {"metrics": ["mt_rag_radbench"]}
        self.assertEqual(evaluator.evaluate(config).metrics, {"mt_rag_radbench": 0.5})
        for selection in ([], ["unknown"], ["mt_rag_faithfulness"] * 2,
                          [{"metric_name": "mt_rag_faithfulness", "unsupported": True}]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                evaluator.evaluate(fixture_config(), metrics_override=selection)

    def test_callback_failure_is_advisory(self):
        def callback(trace):
            raise RuntimeError("Cost model failed")

        with self.assertLogs("rag_stack_evaluator.api", level="WARNING"):
            result = self.evaluator().evaluate(fixture_config(), on_trace_ready=callback)
        self.assertEqual(len(result.metrics), 3)

    def test_missing_duplicate_and_untraced_outputs_fail_before_judge(self):
        def missing(config, requests, recorder, *, options):
            run = run_pipeline(config, requests, recorder, options=options)
            return PipelineRun(run.predictions[:-1], run.workflow)

        def duplicate(config, requests, recorder, *, options):
            run = run_pipeline(config, requests, recorder, options=options)
            return PipelineRun([run.predictions[0]] * 2, run.workflow)

        def untraced(config, requests, recorder, *, options):
            return PipelineRun([Prediction(row["task_id"], "answer", []) for row in requests],
                               {"dataflow_type": "sequential", "stages": []})

        for pipeline in (missing, duplicate, untraced):
            with self.subTest(pipeline=pipeline.__name__), patch("rag_stack_mtrag.backend.score_with_mtrag") as judge:
                with self.assertRaises(ValueError):
                    self.evaluator(pipeline).evaluate(fixture_config())
                judge.assert_not_called()

    def test_configured_task_identity_is_verified_before_running_pipeline(self):
        evaluator = self.evaluator()
        self.task_path.write_text("changed source")
        with patch.object(evaluator, "pipeline") as pipeline, self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            evaluator.evaluate(fixture_config())
        pipeline.assert_not_called()

    def test_installed_entry_point_factory_and_capabilities(self):
        entry = metadata.EntryPoint(name="mtrag", value="rag_stack_mtrag.backend:BACKEND", group="rag_stack.evaluators")
        with patch.dict(factory._BACKENDS, clear=True), patch.object(factory, "_entry_points", return_value=(entry,)):
            self.assertEqual(factory.available_backends(), ("mtrag",))
            evaluator = factory.create_quality_evaluator(backend="mtrag", backend_options=self.options)
            self.assertIsInstance(evaluator, MTRAGEvaluator)
            self.assertEqual(factory.supported_pipeline_modes("mtrag"), ("sequential",))
            self.assertIn("mt_rag_radbench", factory.supported_metrics("mtrag"))
            self.assertFalse(factory.get_backend_spec("mtrag").supports_measured)

    def test_official_wrapper_executes_pinned_full_suite_not_single_metric(self):
        config = self.root / "config.yaml"
        config.write_text("Fixture configuration; no evaluator is run")
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        argv = ["run_official.py", "--repo", str(self.root),
                "--revision", self.options["upstream_revision"], "--provider", "hf",
                "--judge-model", "example-judge", "--algorithmic-config", str(config),
                "--algorithmic-config-sha256", digest,
                "--input", str(self.task_path), "--output", str(self.root / "scored.jsonl")]
        with patch.object(sys, "argv", argv), patch.object(run_official.subprocess, "check_output", return_value=self.options["upstream_revision"]), \
                patch.object(run_official.subprocess, "run"), patch.object(run_official.os, "chdir"), \
                patch.object(run_official.os, "execv") as execute:
            run_official.main()
        command = execute.call_args.args[1]
        self.assertEqual(command[1], str(self.root / "scripts/evaluation/run_generation_eval.py"))
        self.assertIn("--algorithmic_evaluators", command)
        self.assertIn("--judge_model", command)

    def test_example_imports_no_model_libraries(self):
        code = "import sys; import rag_stack_mtrag.backend; assert not {'torch','transformers','ragas','pandas'}.intersection(sys.modules)"
        # A fresh process inherits source paths but deliberately omits the
        # validation guard, whose own eager torch import would mask this check.
        import os
        environment = dict(os.environ, PYTHONPATH=str(EXAMPLE) + ":" + str(EXAMPLE.parents[1]))
        subprocess.run([sys.executable, "-c", code], check=True, env=environment)


if __name__ == "__main__":
    unittest.main()
