import asyncio
import os
import tempfile
import unittest
from unittest import mock


_TEST_PROJECT = tempfile.TemporaryDirectory()
os.environ.setdefault("PROJECT_DIR", _TEST_PROJECT.name)
os.environ.setdefault("FASTLLM_BACKEND_URL", "http://127.0.0.1:8002")

import thinking_proxy


def tearDownModule():
    _TEST_PROJECT.cleanup()


class FakeChild:
    def __init__(self, generation):
        self.generation = generation
        self.returncode = None

class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    async def wait(self):
        return self.returncode


class OwnedFastLLMSpawnTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_spawn_gets_a_private_unique_control_token(self):
        self.assertTrue(
            hasattr(thinking_proxy, "OwnedFastLLMChild"),
            "owned FastLLM child record is not implemented",
        )
        spawned = []

        async def create_process(*argv, **kwargs):
            spawned.append((argv, kwargs))
            return FakeProcess(1000 + len(spawned))

        with tempfile.TemporaryDirectory() as root:
            log_path = os.path.join(root, "owned.log")
            with (
                mock.patch.object(
                    thinking_proxy,
                    "FASTLLM_BACKEND_COMMAND",
                    "/bin/true",
                ),
                mock.patch.object(
                    thinking_proxy,
                    "FASTLLM_BACKEND_CWD",
                    root,
                ),
                mock.patch.object(
                    thinking_proxy,
                    "FASTLLM_BACKEND_LOG",
                    log_path,
                ),
                mock.patch(
                    "thinking_proxy.secrets.token_urlsafe",
                    side_effect=["epoch-one", "epoch-two"],
                ),
                mock.patch(
                    "thinking_proxy.asyncio.create_subprocess_exec",
                    side_effect=create_process,
                ),
            ):
                first = await thinking_proxy._spawn_owned_fastllm(1)
                second = await thinking_proxy._spawn_owned_fastllm(2)

            first_env = spawned[0][1]["env"]
            second_env = spawned[1][1]["env"]
            self.assertEqual(
                first_env["FASTLLM_PREFIX_CACHE_CONTROL_TOKEN"],
                "epoch-one",
            )
            self.assertEqual(
                second_env["FASTLLM_PREFIX_CACHE_CONTROL_TOKEN"],
                "epoch-two",
            )
            self.assertNotEqual(
                first_env["FASTLLM_PREFIX_CACHE_CONTROL_TOKEN"],
                second_env["FASTLLM_PREFIX_CACHE_CONTROL_TOKEN"],
            )
            self.assertNotIn("epoch-one", repr(first))
            self.assertNotIn("epoch-two", repr(second))
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_text = log_file.read()
            self.assertNotIn("epoch-one", log_text)
            self.assertNotIn("epoch-two", log_text)

class PrefixEpochTests(unittest.TestCase):
    def test_backend_epoch_reset_preserves_remote_routing_hints_only(self):
        tracker = thinking_proxy.PrefixTracker(local_slots=2)
        messages = [{"role": "user", "content": "stable prefix"}]
        for backend in ("local", "nim", "or"):
            tracker.record(messages, backend)
        tracker.reset_local()
        self.assertFalse(tracker.hit(messages, "local"))
        self.assertTrue(tracker.hit(messages, "nim"))
        self.assertTrue(tracker.hit(messages, "or"))



class BackendLifecycleStateTests(unittest.IsolatedAsyncioTestCase):
    def make_manager(self, **overrides):
        manager_cls = getattr(thinking_proxy, "BackendLifecycleManager", None)
        self.assertIsNotNone(
            manager_cls,
            "thinking_proxy.BackendLifecycleManager is not implemented",
        )
        calls = overrides.pop("calls", {"start": 0, "stop": 0, "reset": 0})

        async def start_backend(generation):
            calls["start"] += 1
            return FakeChild(generation)

        async def stop_backend(child):
            calls["stop"] += 1
            child.returncode = 0

        async def probe_ready(child):
            return True

        def reset_local():
            calls["reset"] += 1

        options = {
            "owned": True,
            "start_backend": start_backend,
            "stop_backend": stop_backend,
            "probe_ready": probe_ready,
            "reset_local": reset_local,
            "start_timeout": 1.0,
            "idle_timeout": 0.0,
            "minimum_free_bytes": 0,
            "resume_free_bytes": 0,
        }
        options.update(overrides)
        return manager_cls(**options), calls

    async def test_two_cold_callers_share_one_activation(self):
        entered = asyncio.Event()
        allow_start = asyncio.Event()
        calls = {"start": 0, "stop": 0, "reset": 0}

        async def start_backend(generation):
            calls["start"] += 1
            entered.set()
            await allow_start.wait()
            return FakeChild(generation)

        manager, _ = self.make_manager(
            calls=calls,
            start_backend=start_backend,
        )
        first = asyncio.create_task(manager.acquire())
        second = asyncio.create_task(manager.acquire())
        await entered.wait()
        await asyncio.sleep(0)
        self.assertEqual(calls["start"], 1)
        self.assertEqual(manager.state, "STARTING")

        allow_start.set()
        first_lease, second_lease = await asyncio.gather(first, second)
        self.assertEqual(manager.state, "READY")
        self.assertEqual(manager.active, 2)
        await first_lease.release()
        await second_lease.release()
        self.assertEqual(manager.active, 0)

    async def test_start_failure_is_shared_by_all_waiters(self):
        calls = {"start": 0, "stop": 0, "reset": 0}

        async def fail_start(generation):
            calls["start"] += 1
            await asyncio.sleep(0)
            raise RuntimeError("load failed")

        manager, _ = self.make_manager(calls=calls, start_backend=fail_start)
        results = await asyncio.gather(
            manager.acquire(), manager.acquire(), return_exceptions=True)
        self.assertEqual(calls["start"], 1)
        self.assertEqual(manager.state, "FAILED")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(item, thinking_proxy.BackendLifecycleError)
                            for item in results))
        self.assertTrue(all("load failed" in str(item) for item in results))

    async def test_idle_stop_then_next_request_starts_new_generation(self):
        manager, calls = self.make_manager(idle_timeout=10.0)
        first = await manager.acquire()
        first_generation = manager.generation
        await first.release()
        self.assertFalse(await manager.check_idle(manager.last_idle_at + 9.0))
        self.assertTrue(await manager.check_idle(manager.last_idle_at + 10.0))
        self.assertEqual(manager.state, "COLD")
        self.assertEqual(calls["stop"], 1)

        second = await manager.acquire()
        self.assertGreater(manager.generation, first_generation)
        self.assertEqual(calls["start"], 2)
        await second.release()

    async def test_pressure_waits_for_last_lease_and_hysteresis_blocks_restart(self):
        manager, calls = self.make_manager(
            minimum_free_bytes=20,
            resume_free_bytes=35,
        )
        lease = await manager.acquire()
        await manager.observe_memory(free_bytes=10, total_bytes=100)
        self.assertEqual(manager.state, "DRAINING")
        self.assertEqual(calls["stop"], 0)
        with self.assertRaises(thinking_proxy.BackendMemoryPressure):
            await manager.acquire()

        await lease.release()
        self.assertEqual(manager.state, "COLD")
        self.assertEqual(calls["stop"], 1)
        with self.assertRaises(thinking_proxy.BackendMemoryPressure):
            await manager.acquire()

        await manager.observe_memory(free_bytes=35, total_bytes=100)
        restarted = await manager.acquire()
        self.assertEqual(calls["start"], 2)
        await restarted.release()

    async def test_external_backend_is_never_stopped_or_signalled(self):
        calls = {"start": 0, "stop": 0, "reset": 0}
        manager, _ = self.make_manager(calls=calls, owned=False)
        lease = await manager.acquire()
        await lease.release()
        await manager.stop("shutdown")
        self.assertEqual(calls["start"], 0)
        self.assertEqual(calls["stop"], 0)

    async def test_each_owned_backend_epoch_resets_local_cache_expectations(self):
        manager, calls = self.make_manager(idle_timeout=1.0)
        first = await manager.acquire()
        await first.release()
        await manager.check_idle(manager.last_idle_at + 1.0)
        second = await manager.acquire()
        await second.release()
        self.assertEqual(calls["start"], 2)
        self.assertEqual(calls["reset"], 2)

    async def test_ratio_watermark_uses_hysteresis(self):
        manager, calls = self.make_manager(
            high_used_ratio=0.80,
            resume_used_ratio=0.60,
        )
        lease = await manager.acquire()
        await manager.observe_memory(free_bytes=19, total_bytes=100)
        self.assertEqual(manager.state, "DRAINING")
        await lease.release()
        self.assertEqual(calls["stop"], 1)

        await manager.observe_memory(free_bytes=30, total_bytes=100)
        with self.assertRaises(thinking_proxy.BackendMemoryPressure):
            await manager.acquire()
        await manager.observe_memory(free_bytes=40, total_bytes=100)
        restarted = await manager.acquire()
        await restarted.release()

    async def test_timed_out_activation_waiter_does_not_create_a_lease(self):
        entered = asyncio.Event()
        allow_start = asyncio.Event()

        async def slow_start(generation):
            entered.set()
            await allow_start.wait()
            return FakeChild(generation)

        manager, _ = self.make_manager(start_backend=slow_start)
        waiter = asyncio.create_task(manager.acquire(timeout=0.01))
        await entered.wait()
        with self.assertRaises(asyncio.TimeoutError):
            await waiter
        self.assertEqual(manager.active, 0)

        allow_start.set()
        await manager.wait_for_activation()
        self.assertEqual(manager.active, 0)
        lease = await manager.acquire()
        self.assertEqual(manager.active, 1)
        await lease.release()
    async def test_shutdown_during_start_cancels_and_reaps_child(self):
        probing = asyncio.Event()
        allow_ready = asyncio.Event()

        async def probe_ready(child):
            probing.set()
            await allow_ready.wait()
            return True

        manager, calls = self.make_manager(probe_ready=probe_ready)
        acquire = asyncio.create_task(manager.acquire())
        await probing.wait()
        self.assertTrue(await manager.stop("shutdown"))
        result = await asyncio.gather(acquire, return_exceptions=True)
        self.assertIsInstance(result[0], asyncio.CancelledError)
        self.assertEqual(calls["stop"], 1)
        self.assertEqual(manager.state, "COLD")
        self.assertIsNone(manager.child)

    async def test_owned_stop_checkpoints_after_drain_before_signal(self):
        events = []

        async def checkpoint_backend(child):
            events.append(("checkpoint", child.generation))
            return {
                "generation": 7,
                "pages": 11,
                "bytes": 4096,
                "duration_ms": 3.5,
            }

        async def stop_backend(child):
            events.append(("stop", child.generation))
            child.returncode = 0

        manager, _ = self.make_manager(
            checkpoint_backend=checkpoint_backend,
            stop_backend=stop_backend,
        )
        lease = await manager.acquire()
        shutdown = asyncio.create_task(
            manager.drain_and_stop("proxy_shutdown", timeout=1.0))
        await asyncio.sleep(0)
        self.assertEqual(events, [])
        await lease.release()
        self.assertTrue(await shutdown)
        self.assertEqual(events, [("checkpoint", 1), ("stop", 1)])
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["checkpoint_successes"], 1)
        self.assertEqual(snapshot["last_checkpoint_generation"], 7)
        self.assertIsNone(snapshot["last_checkpoint_error"])

    async def test_checkpoint_failure_is_recorded_but_owned_stop_continues(self):
        events = []

        async def checkpoint_backend(child):
            events.append("checkpoint")
            raise RuntimeError("checkpoint boom")

        async def stop_backend(child):
            events.append("stop")
            child.returncode = 0

        manager, _ = self.make_manager(
            checkpoint_backend=checkpoint_backend,
            stop_backend=stop_backend,
        )
        lease = await manager.acquire()
        await lease.release()
        self.assertTrue(await manager.stop("idle"))
        self.assertEqual(events, ["checkpoint", "stop"])
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["checkpoint_failures"], 1)
        self.assertIn(
            "checkpoint boom", snapshot["last_checkpoint_error"])

    async def test_forced_stop_with_active_lease_skips_checkpoint(self):
        checkpoints = 0

        async def checkpoint_backend(child):
            nonlocal checkpoints
            checkpoints += 1

        manager, calls = self.make_manager(
            checkpoint_backend=checkpoint_backend)
        lease = await manager.acquire()
        self.assertTrue(await manager.stop("timeout", force=True))
        self.assertEqual(checkpoints, 0)
        self.assertEqual(calls["stop"], 1)
        await lease.release()


    async def test_shutdown_drains_active_lease_before_stop(self):
        manager, calls = self.make_manager()
        lease = await manager.acquire()
        shutdown = asyncio.create_task(
            manager.drain_and_stop("proxy_shutdown", timeout=1.0))
        await asyncio.sleep(0)
        self.assertEqual(manager.state, "DRAINING")
        with self.assertRaises(thinking_proxy.BackendLifecycleError):
            await manager.acquire()
        self.assertEqual(calls["stop"], 0)
        await lease.release()
        self.assertTrue(await shutdown)
        self.assertEqual(calls["stop"], 1)
        self.assertEqual(manager.state, "COLD")

    async def test_shutdown_timeout_forces_owned_child_stop(self):
        manager, calls = self.make_manager()
        lease = await manager.acquire()
        self.assertTrue(
            await manager.drain_and_stop("proxy_shutdown", timeout=0.01))
        self.assertEqual(calls["stop"], 1)
        self.assertEqual(manager.state, "COLD")
        await lease.release()

    async def test_timed_out_queue_item_is_not_sent_after_cold_activation(self):
        entered = asyncio.Event()
        allow_ready = asyncio.Event()
        sent = 0

        class Lease:
            async def release(self):
                pass

        class Lifecycle:
            async def acquire(self, timeout=None):
                entered.set()
                await allow_ready.wait()
                return Lease()

        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                pass

            async def post(self, url, json):
                nonlocal sent
                sent += 1
                return thinking_proxy.httpx.Response(
                    200,
                    json={"choices": [], "usage": {}},
                    request=thinking_proxy.httpx.Request("POST", url),
                )

        scheduler = thinking_proxy.BackendScheduler(max_concurrent=1)
        with (
            mock.patch.object(thinking_proxy, "backend_lifecycle", Lifecycle()),
            mock.patch.object(thinking_proxy.httpx, "AsyncClient", Client),
        ):
            await scheduler.start_workers()
            try:
                with self.assertRaises(asyncio.TimeoutError):
                    await scheduler.submit(
                        {"model": "qwen3.6-fastllm", "messages": []},
                        priority=0,
                        timeout=0.01,
                    )
                await entered.wait()
                allow_ready.set()
                await asyncio.sleep(0.02)
                self.assertEqual(sent, 0)
            finally:
                for worker in scheduler._workers:
                    worker.cancel()
                await asyncio.gather(*scheduler._workers, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
