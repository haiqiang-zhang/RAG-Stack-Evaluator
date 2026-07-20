from types import SimpleNamespace

import pandas as pd

from rag_stack.static_rag_evaluator.static_rag_evaluator import (
	StaticRAGEvaluatorQualityOnly,
)


def test_react_uses_resolved_generator_component(monkeypatch):
	created = {}
	result = object()

	class FakeRetriever:
		def __init__(self, project_dir, vectordb):
			created["retriever"] = (project_dir, vectordb)

	class FakeVllmAPI:
		def __init__(self, project_dir, **kwargs):
			created["generator"] = (project_dir, kwargs)

	def fake_run_react(**kwargs):
		created["run_react"] = kwargs
		return result

	import rag_stack.static_rag_evaluator.agentic_react as react_module
	import rag_stack.static_rag_evaluator.nodes.semanticretrieval.vectordb as vdb_module

	monkeypatch.setattr(react_module, "run_react", fake_run_react)
	monkeypatch.setattr(vdb_module, "VectorDB", FakeRetriever)

	retrieval_node = SimpleNamespace(
		stage="semantic_retrieval",
		node_params={"top_k": 7},
		module=SimpleNamespace(
			component="vectordb",
			module_param={"vectordb": "shared", "nprobe": 4},
		),
	)
	generator_node = SimpleNamespace(
		stage="generator",
		node_params={},
		module=SimpleNamespace(
			component="vllm_api",
			module=FakeVllmAPI,
			module_param={
				"model": "Qwen/Qwen2.5-14B-Instruct",
				"uri": "http://judge:8000",
				"max_tokens": 512,
				"temperature": 0.3,
			},
		),
	)
	evaluator = SimpleNamespace(
		project_dir="/tmp/project",
		qa_data=pd.DataFrame({"query": ["question"]}),
	)

	actual = StaticRAGEvaluatorQualityOnly._run_react(
		evaluator,
		{"pipeline_runtime": {"max_iter": 3}},
		{"line": [retrieval_node, generator_node]},
		[{"name": "shared", "embedding_model": "mpnet"}],
	)

	assert actual is result
	assert created["retriever"] == ("/tmp/project", "shared")
	project_dir, generator_kwargs = created["generator"]
	assert project_dir == "/tmp/project"
	assert generator_kwargs["model"] == "Qwen/Qwen2.5-14B-Instruct"
	assert generator_kwargs["uri"] == "http://judge:8000"
	assert generator_kwargs["max_tokens"] == 512
	assert created["run_react"]["generator_model"] == (
		"Qwen/Qwen2.5-14B-Instruct"
	)
	assert created["run_react"]["max_iter"] == 3
	assert created["run_react"]["gen_params"] == {
		"temperature": 0.3,
		"max_tokens": 512,
	}
