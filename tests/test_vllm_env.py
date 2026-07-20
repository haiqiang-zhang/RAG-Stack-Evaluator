import os
import sys

from rag_stack.static_rag_evaluator.measured.vllm_env import configure_vllm_worker_env


def test_configure_vllm_worker_env_adds_env_bin_and_removes_hpcx_ucx():
	env = {
		"PATH": os.pathsep.join(["/usr/bin", "/bin"]),
		"LD_LIBRARY_PATH": os.pathsep.join(
			[
				"/opt/hpcx/ucx/lib",
				"/usr/local/ucx/lib",
				"/usr/lib",
				"/opt/hpcx/ompi/lib",
			]
		),
	}

	configure_vllm_worker_env(env=env)

	path_parts = env["PATH"].split(os.pathsep)
	assert os.path.join(sys.prefix, "bin") in path_parts[:3]
	assert "/opt/hpcx/ucx/lib" not in env["LD_LIBRARY_PATH"].split(os.pathsep)
	assert "/usr/local/ucx/lib" not in env["LD_LIBRARY_PATH"].split(os.pathsep)
	assert "/opt/hpcx/ompi/lib" in env["LD_LIBRARY_PATH"].split(os.pathsep)
