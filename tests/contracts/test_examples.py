"""Exercise shipped examples through the real public adapter, without models."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rag_stack.evaluation import api, registry
from rag_stack_evaluator import backends

EXAMPLES = Path(__file__).resolve().parents[2] / 'example'


def example(name):
    spec = importlib.util.spec_from_file_location('example_' + name, EXAMPLES / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def concrete_runtime(monkeypatch):
    trace = {'queries': [{'question_id': 'q1', 'calls': [{
        'stage': 'generator', 'input_tokens': 8, 'output_tokens': 3, 'step_idx': 0,
    }]}]}
    state = SimpleNamespace(trace=trace, engines=[], datasets=[], measured=[])

    class Dataset:
        def __init__(self, **kwargs):
            state.datasets.append(kwargs)

    class Engine:
        def __init__(self, dataset=None, project_dir=None, *, dataset_manager=None):
            self.manager = dataset_manager
            state.engines.append(self)

        def _run_pipeline(self, *args, **kwargs):
            raise AssertionError('Fake provider must not run model operations')

        def evaluate(self, config, run_dir=None, metrics_override=None, on_trace_ready=None):
            self.call = (config, run_dir, metrics_override)
            if on_trace_ready:
                on_trace_ready(trace)
            return {'f1': .75, '__execution_dag__': trace}

    class Provider:
        def __init__(self, engine, gpus, **kwargs):
            assert callable(engine._run_pipeline), 'MeasuredProvider needs static engine execution capabilities'
            self.engine = engine
            state.measured.append((gpus, kwargs))
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            state.closed = True
            self.entered = False

        def evaluate(self, config, system, **kwargs):
            assert self.entered
            state.measured_call = (config, system, kwargs)
            return SimpleNamespace(performance_score=2.5, quality={'f1': .75},
                                   raw_performance={'qps': 2.5}, performance_execution_trace=trace)

    monkeypatch.setitem(sys.modules, 'rag_stack_evaluator.static_rag_evaluator', SimpleNamespace(
        DatasetEvalManager=Dataset, StaticRAGEvaluatorQualityOnly=Engine, MeasuredProvider=Provider,
    ))
    monkeypatch.setattr(registry, '_BACKENDS', {'static_gt': backends.STATIC_GT})
    monkeypatch.setattr(registry, '_entry_points', lambda: ())
    return state


@pytest.mark.parametrize('script,function,extras', [
    ('quality', 'evaluate_quality', {}),
    ('quality_metrics_override', 'evaluate_selected_metrics', {'metrics': ['f1']}),
    ('quality_trace', 'evaluate_with_trace', {}),
])
def test_quality_examples_keep_data_separate_and_return_public_result(concrete_runtime, script, function, extras):
    module = example(script)
    algo = json.loads((EXAMPLES / 'algo_config.json').read_text())
    original = deepcopy(algo)
    seen = []
    if script == 'quality_trace':
        extras = {'on_trace_ready': seen.append}
    result = getattr(module, function)(pipeline_config=algo, qa_parquet='caller-qa.parquet',
        corpus_parquet='caller-chunks.parquet', project_dir='project', run_dir='run', **extras)
    assert isinstance(result, api.EvaluationResult)
    assert result.metrics == {'f1': .75}
    assert result.trace is concrete_runtime.trace
    if seen:
        assert seen[0] is result.trace
    assert concrete_runtime.datasets == [{'project_dir': 'project', 'qa_data_path': 'caller-qa.parquet',
                                          'corpus_data_path': 'caller-chunks.parquet'}]
    assert concrete_runtime.engines[0].call[0] is algo
    assert algo == original and 'dataset' not in algo
    assert concrete_runtime.engines[0].call[2] == extras.get('metrics')


def test_measured_example_keeps_static_provider_contract(concrete_runtime):
    module = example('measured')
    algo = json.loads((EXAMPLES / 'algo_config.json').read_text())
    system = json.loads((EXAMPLES / 'system_config.json').read_text())
    result = module.evaluate_measured(pipeline_config=algo, system_config=system,
        available_gpus=['cuda:0'], qa_parquet='qa.parquet', corpus_parquet='chunks.parquet',
        project_dir='project', run_dir='run', n_queries=3, selection='max_throughput',
        force_disagg=False, require_admissible=True)
    assert result == {'performance_score': 2.5, 'quality': {'f1': .75},
                      'raw_performance': {'qps': 2.5}, 'performance_execution_trace': concrete_runtime.trace}
    assert concrete_runtime.measured_call[:2] == (algo, system)
    assert concrete_runtime.closed is True
    assert 'config' not in concrete_runtime.datasets[0]
    assert 'dataset' not in algo and 'dataset' not in system


def test_shipped_config_pair_uses_real_shared_stage_and_placement_contracts():
    from rag_stack.system_layout import cm_fixed_placement_inputs, engine_parallelism
    algo = json.loads((EXAMPLES / 'algo_config.json').read_text())
    system = json.loads((EXAMPLES / 'system_config.json').read_text())
    groups, chips, servers, stages = cm_fixed_placement_inputs(system, algo)
    assert stages == ['generator_prefill', 'generator_decode']
    assert groups == [['generator_prefill', 'generator_decode']]
    assert chips == {'generator_prefill': 1, 'generator_decode': 1}
    assert servers == 1
    assert engine_parallelism(system, 'generator') == (1, 1)


@pytest.mark.parametrize('script', ['quality', 'quality_metrics_override', 'quality_trace', 'measured'])
def test_example_help_needs_no_model_runtime(script):
    result = subprocess.run([sys.executable, str(EXAMPLES / f'{script}.py'), '--help'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--algo-config' in result.stdout and '--qa' in result.stdout and '--corpus' in result.stdout


def test_real_measured_wrapper_uses_static_adapter_pipeline_not_typed_evaluate():
    import pandas as pd
    from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import MeasuredEvaluator
    from rag_stack_evaluator.static_rag_evaluator.measured.performance_trace import make_performance_trace_envelope
    calls = [{'stage': 'generator', 'input_tokens': 8, 'output_tokens': 3, 'step_idx': 0}]
    performance_trace = make_performance_trace_envelope([calls], invocation_ids=['request-1'],
        source_question_ids=['0'], capacity=1, population_queries=1)
    seen = []
    class StaticEngine:
        def evaluate(self, *args, **kwargs):
            raise AssertionError('Measured execution must not call public quality evaluate')
        def _run_pipeline(self, config, **kwargs):
            seen.append(config)
            return SimpleNamespace(sequential_perf={'qps': 2.5, '__performance_execution_trace__': performance_trace},
                quality={'f1': .75}, trace=[calls], n_evaluated=1,
                previous_result=pd.DataFrame({'__qid__': ['request-1']}))
    adapter = backends._EngineAdapter(StaticEngine())
    algo = json.loads((EXAMPLES / 'algo_config.json').read_text())
    result = MeasuredEvaluator(adapter).evaluate(algo, cache=object(), system_config={}, generation_defaults_applied=True)
    assert seen == [algo]
    assert result['quality']['f1'] == .75
    assert result['quality']['__execution_dag__']['queries'][0]['question_id'] == '0'
    assert result['performance_execution_trace'] is performance_trace
