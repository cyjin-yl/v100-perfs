import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "benchmarks" / "agent_bench.py"
SPEC = importlib.util.spec_from_file_location("agent_bench_under_test", MODULE_PATH)
agent_bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent_bench
SPEC.loader.exec_module(agent_bench)


class RunAgentsFailureTest(unittest.TestCase):
    def test_worker_failure_is_raised_to_caller(self):
        original = agent_bench.run_one_turn

        def fail_worker(*args, **kwargs):
            raise RuntimeError("request failed")

        agent_bench.run_one_turn = fail_worker
        try:
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                agent_bench.run_agents(
                    endpoint="http://127.0.0.1:1",
                    model="test-model",
                    num_agents=2,
                    max_model_len=1024,
                    shared_prefix_tokens=1,
                    turns_per_agent=1,
                    max_tokens=1,
                )
        finally:
            agent_bench.run_one_turn = original


if __name__ == "__main__":
    unittest.main()
