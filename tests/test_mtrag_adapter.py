"""CPU contract tests. The subprocess scorer is a synthetic fixture, not MT-RAG."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


MODULE = Path(__file__).resolve().parents[1] / "rag_stack_evaluator/integrations/mtrag.py"
SPEC = importlib.util.spec_from_file_location("mtrag_contract_test", MODULE)
mtrag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mtrag)

REVISION = "53fef6f5917c7e10c436f697e69a955dbc795d81"
SCORER_FIXTURE = '''
import json, pathlib, sys, time
source, destination, mode = sys.argv[1:]
rows = [json.loads(line) for line in pathlib.Path(source).read_text().splitlines()]
for index, row in enumerate(rows):
    row["metrics"] = {"RL_F": [0.25 + index * 0.5], "Length": [len(row["predictions"][0]["text"])]}
if mode == "exit": sys.exit(7)
if mode == "timeout": time.sleep(10)
if mode == "missing": rows.pop()
if mode == "duplicate": rows.append(rows[0])
if mode == "extra":
    other = dict(rows[0], task_id="unexpected-task")
    rows.append(other)
if mode in ("input", "contexts", "predictions", "targets"): rows[0][mode] = []
if mode == "scalar": rows[0]["metrics"]["RL_F"] = 0.5
if mode == "missing_metric": rows[0]["metrics"] = {}
if mode == "long_list": rows[0]["metrics"]["RL_F"] = [0.5, 0.5]
if mode.startswith("score:"):
    rows[0]["metrics"]["RL_F"] = [json.loads(mode.split(":", 1)[1])]
pathlib.Path(destination).write_text("".join(json.dumps(row) + "\\n" for row in reversed(rows)))
'''


def fixture_tasks():
    return [{
        "task_id": f"conversation<::>{index}", "conversation_id": "conversation",
        "input": [{"speaker": "user", "text": "Earlier question?"},
                  {"speaker": "agent", "text": "Earlier benchmark answer."},
                  {"speaker": "user", "text": f"Follow-up {index}?"}],
        "targets": [{"speaker": "agent", "text": f"Reference {index}"}],
        "contexts": [{"document_id": "original-doc", "text": "Original context", "score": 9}],
        "predictions": [{"text": "Stale output"}], "metrics": {"RL_F": [1.0]},
    } for index in (1, 2)]


def fixture_inputs(tasks):
    return [SimpleNamespace(qid=row["qid"], query=row["query"],
                            generated_texts=f"Actual answer {index}",
                            retrieved_contents=[f"Actual context {index}"])
            for index, row in enumerate(mtrag.to_qa_rows(tasks))]


class MTRagAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mtrag_contract_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tasks = fixture_tasks()
        self.inputs = fixture_inputs(self.tasks)
        self.tasks_path = self.root / "tasks.jsonl"
        self.tasks_path.write_text("".join(json.dumps(task) + "\n" for task in self.tasks))
        self.digest = hashlib.sha256(self.tasks_path.read_bytes()).hexdigest()
        self.scorer = self.root / "contract fixture; no shell.py"
        self.scorer.write_text(SCORER_FIXTURE)

    def score(self, mode="ok", **overrides):
        options = dict(tasks_path=str(self.tasks_path), tasks_sha256=self.digest,
                       scorer_command=[sys.executable, str(self.scorer), "{input}", "{output}", mode],
                       scorer_identity="synthetic-contract-fixture-v1", upstream_revision=REVISION)
        options.update(overrides)
        return mtrag.mt_rag_faithfulness(self.inputs, **options)

    def test_hash_verified_task_loading(self):
        self.assertEqual(mtrag.load_tasks(self.tasks_path, self.digest), self.tasks)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            mtrag.load_tasks(self.tasks_path, "0" * 64)
        with self.assertRaisesRegex(ValueError, "full SHA-256"):
            mtrag.load_tasks(self.tasks_path, "short")

    def test_duplicate_and_missing_original_task_ids_rejected(self):
        for bad in (self.tasks + [self.tasks[0]], [dict(self.tasks[0], task_id=None)]):
            with self.subTest(tasks=bad), self.assertRaisesRegex(ValueError, "task_id"):
                mtrag.to_qa_rows(bad)

    def test_malformed_conversations_and_targets_rejected(self):
        for field, value in (("input", []), ("input", [{"text": "no speaker"}]),
                             ("targets", []), ("targets", [{"text": None}])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                mtrag.to_qa_rows([dict(self.tasks[0], **{field: value})])

    def test_flattening_matches_upstream_and_targets_never_enter_query(self):
        rows = mtrag.to_qa_rows(self.tasks)
        self.assertEqual(set(rows[0]), {"qid", "query", "generation_gt"})
        self.assertEqual(rows[0]["query"], "user: Earlier question?\nagent: Earlier benchmark answer.\nuser: Follow-up 1?")
        self.assertEqual(rows[0]["generation_gt"], ["Reference 1"])
        self.assertNotIn("Reference 1", rows[0]["query"])

    def test_actual_outputs_replace_reference_contexts_without_source_mutation(self):
        before = deepcopy(self.tasks)
        rows = mtrag.build_judge_records(self.inputs, self.tasks)
        self.assertEqual(self.tasks, before)
        self.assertEqual(rows[0]["contexts"], [{"text": "Actual context 0"}])
        self.assertEqual(rows[0]["predictions"], [{"text": "Actual answer 0"}])
        self.assertNotIn("metrics", rows[0])
        rows[0]["input"][0]["text"] = "Changed copy"
        self.assertEqual(self.tasks, before)

    def test_qid_join_survives_row_reordering_and_query_normalization(self):
        self.inputs[0].query = "Host-normalized query"
        rows = mtrag.build_judge_records(self.inputs[::-1], self.tasks)
        self.assertEqual([row["task_id"] for row in rows], [item.qid for item in self.inputs[::-1]])
        self.assertEqual(rows[1]["input"], self.tasks[0]["input"])

    def test_task_subset_is_supported_without_scoring_unrequested_tasks(self):
        rows = mtrag.build_judge_records(self.inputs[:1], self.tasks)
        self.assertEqual(len(rows), 1)

    def test_empty_actual_answer_and_retrieval_remain_real_outcomes(self):
        self.inputs[0].generated_texts = ""
        self.inputs[0].retrieved_contents = []
        rows = mtrag.build_judge_records(self.inputs, self.tasks)
        self.assertEqual(rows[0]["predictions"], [{"text": ""}])
        self.assertEqual(rows[0]["contexts"], [])

    def test_missing_actual_outputs_fail_before_scorer(self):
        for field, value in (("generated_texts", None), ("retrieved_contents", None),
                             ("retrieved_contents", [None])):
            items = deepcopy(self.inputs)
            setattr(items[0], field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                mtrag.build_judge_records(items, self.tasks)

    def test_missing_unknown_and_duplicate_metric_ids_fail(self):
        for qid in (None, "unknown", self.inputs[1].qid):
            items = deepcopy(self.inputs)
            items[0].qid = qid
            with self.subTest(qid=qid), self.assertRaisesRegex(ValueError, "qid"):
                mtrag.build_judge_records(items, self.tasks)

    def test_empty_batches_fail(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            mtrag.build_judge_records([], self.tasks)

    def test_scores_are_aligned_by_id_not_output_order(self):
        self.assertEqual(self.score(), [0.25, 0.75])

    def test_incomplete_extra_duplicate_and_changed_records_fail(self):
        for mode in ("missing", "duplicate", "extra", "input", "contexts", "predictions", "targets"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.score(mode)

    def test_invalid_score_shapes_and_values_fail(self):
        modes = ["scalar", "missing_metric", "long_list"]
        modes += ["score:" + value for value in ("true", "null", '"0.5"', "NaN", "Infinity", "-0.1", "1.1")]
        for mode in modes:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.score(mode)

    def test_subprocess_failure_propagates(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.score("exit")

    def test_full_evaluator_transport_returns_multiple_selected_metrics(self):
        scores = mtrag.score_with_mtrag(
            self.inputs, tasks_path=str(self.tasks_path), tasks_sha256=self.digest,
            scorer_command=[sys.executable, str(self.scorer), "{input}", "{output}", "ok"],
            scorer_identity="synthetic-contract-fixture-v1", upstream_revision=REVISION,
            metric_keys={"faithfulness": "RL_F", "response_length": "Length"},
        )
        self.assertEqual(scores["faithfulness"], [0.25, 0.75])
        self.assertEqual(scores["response_length"], [len(item.generated_texts) for item in self.inputs])

    def test_generic_metrics_require_explicit_ranges_and_singleton_scores(self):
        options = dict(tasks_path=str(self.tasks_path), tasks_sha256=self.digest,
                       scorer_command=[sys.executable, str(self.scorer), "{input}", "{output}", "score:-2"],
                       scorer_identity="synthetic-contract-fixture-v1", upstream_revision=REVISION,
                       metric_keys={"generic": "RL_F"})
        self.assertEqual(mtrag.score_with_mtrag(self.inputs, **options)["generic"], [-2, 0.75])
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            mtrag.score_with_mtrag(self.inputs, **options, score_bounds={"generic": (0, 1)})
        for key in ("missing",):
            with self.assertRaisesRegex(ValueError, "Expected metrics"):
                mtrag.score_with_mtrag(self.inputs, **dict(options, metric_keys={"generic": key}))

    def test_invalid_generic_metric_options_fail_before_subprocess(self):
        options = dict(tasks_path=str(self.tasks_path), tasks_sha256=self.digest,
                       scorer_command=[sys.executable, str(self.scorer), "{input}", "{output}", "ok"],
                       scorer_identity="synthetic-contract-fixture-v1", upstream_revision=REVISION,
                       metric_keys={"generic": "RL_F"})
        invalid = [{"metric_keys": {}}, {"metric_keys": {"__private": "RL_F"}},
                   {"metric_keys": {"public": ""}}, {"score_bounds": {"unknown": (0, 1)}},
                   {"score_bounds": {"generic": (1, 0)}}, {"score_bounds": {"generic": (False, 1)}},
                   {"score_bounds": {"generic": (0, float("inf"))}}]
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                mtrag.score_with_mtrag(self.inputs, **dict(options, **update))

    def test_timeout_propagates(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.score("timeout", timeout_seconds=0.05)

    def test_command_and_provenance_validation(self):
        invalid_options = [{"scorer_identity": ""}, {"upstream_revision": "main"},
                           {"timeout_seconds": 0}, {"timeout_seconds": True},
                           {"scorer_command": "python scorer.py"},
                           {"scorer_command": ["python", "--input={input}", "{output}"]},
                           {"scorer_command": ["python", "{input}", "{input}", "{output}"]}]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.score(**options)

    def test_audit_keeps_each_success_and_failure_separately(self):
        audit = self.root / "audit"
        self.score(audit_dir=str(audit))
        with self.assertRaises(ValueError):
            self.score("missing", audit_dir=str(audit))
        directories = list(audit.iterdir())
        self.assertEqual(len(directories), 2)
        provenance = [json.loads((directory / "provenance.json").read_text()) for directory in directories]
        self.assertEqual({record["status"] for record in provenance}, {"complete", "failed"})
        self.assertTrue(all(record["tasks_sha256"] == self.digest for record in provenance))
        for record in provenance:
            self.assertEqual(record["task_ids"], [item.qid for item in self.inputs])
            self.assertEqual(record["timeout_seconds"], 3600)
            self.assertEqual(record["scorer_command"][:4],
                             [sys.executable, str(self.scorer), "{input}", "{output}"])
            self.assertEqual(record["scorer_command"][4],
                             "ok" if record["status"] == "complete" else "missing")
        for directory in directories:
            for filename in ("input.jsonl", "output.jsonl", "stdout.log", "stderr.log"):
                self.assertTrue((directory / filename).is_file())

    def test_module_import_does_not_load_model_libraries(self):
        code = "import sys; sys.path.insert(0,sys.argv[1]); import rag_stack_evaluator.integrations.mtrag; assert not {'torch','transformers','ragas','pandas'}.intersection(sys.modules)"
        subprocess.run([sys.executable, "-c", code, str(MODULE.parents[2])], check=True)


if __name__ == "__main__":
    unittest.main()
