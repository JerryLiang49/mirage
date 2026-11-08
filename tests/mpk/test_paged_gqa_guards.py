"""CPU-only public guard regressions. Run with python3 -m unittest discover -s tests/mpk."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "python/mirage/mpk" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bindings = load_module("paged_gqa_bindings")
contract = load_module("paged_gqa_contract")


class GuardTests(unittest.TestCase):
    def test_external_stream_launch_never_reaches_kernel(self):
        calls = []
        kernel = SimpleNamespace(launch_on_stream=lambda stream: calls.append(stream))
        pk = SimpleNamespace(_paged_gqa_contract={})
        guarded = bindings.GuardedGQA(kernel, pk, {}, None)
        for closed, inflight in ((False, False), (False, True), (True, False)):
            guarded.closed, guarded.inflight = closed, inflight
            with self.subTest(closed=closed, inflight=inflight):
                with self.assertRaisesRegex(RuntimeError, "external launch_on_stream"):
                    guarded.launch_on_stream(object())
                self.assertEqual(calls, [])
                self.assertEqual(guarded.inflight, inflight)

    def test_blackwell_paths_reject_other_architectures(self):
        for record in ({"implementation": "sm100"},):
            contract.validate_device(record, (10, 0))
            for capability in ((8, 0), (9, 0), (10, 3), (12, 0)):
                with self.subTest(record=record, capability=capability):
                    with self.assertRaisesRegex(ValueError, "requires SM100"):
                        contract.validate_device(record, capability)

    def test_portable_and_direct_are_rejected(self):
        for record in ({"implementation":"portable"}, {"implementation":"sm100","mode":"direct"}):
            with self.assertRaisesRegex(ValueError, "only partitioned sm100"):
                contract.validate_device(record, (10, 0))


class LifecycleTests(unittest.TestCase):
    """Exercise ownership transitions with injected asynchronous failures.

    CUDA and descriptor validation are mocked here. These tests establish host
    state-machine behavior, not GPU completion or numerical correctness.
    """
    def setUp(self):
        self.kernel = Mock()
        self.lease = Mock()
        self.pk = SimpleNamespace(_paged_gqa_contract={})
        self.guard = bindings.GuardedGQA(self.kernel, self.pk, {}, self.lease)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
        self.validation = patch.object(bindings, "validate_bound")
        self.validation.start()
        self.addCleanup(self.validation.stop)
        self.modules = patch.dict("sys.modules", {"torch": self.torch})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_double_launch_rejected_and_finalize_waits(self):
        order = []
        self.kernel.wait.side_effect = lambda: order.append("wait")
        self.kernel.finalize.side_effect = lambda: order.append("finalize")
        self.lease.close.side_effect = lambda: order.append("release")
        self.guard.launch()
        with self.assertRaisesRegex(RuntimeError, "wait/done required"):
            self.guard.launch()
        self.kernel.launch.assert_called_once()
        self.guard.finalize()
        self.assertEqual(order, ["wait", "finalize", "release"])
        self.guard.finalize()
        self.assertEqual(len(order), 3)
        with self.assertRaisesRegex(RuntimeError, "finalized"):
            self.guard.launch()

    def test_failed_launch_requires_device_sync_not_stale_event(self):
        self.kernel.launch.side_effect = RuntimeError("enqueue failed")
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            self.guard.launch()
        self.assertTrue(self.guard.inflight)
        self.assertFalse(self.guard.done())
        self.torch.cuda.synchronize.reset_mock()
        self.guard.wait()
        self.torch.cuda.synchronize.assert_called_once_with(0)
        self.kernel.wait.assert_not_called()
        self.assertFalse(self.guard.inflight)

    def test_failed_recovery_does_not_release_storage(self):
        self.kernel.launch.side_effect = RuntimeError("enqueue failed")
        with self.assertRaises(RuntimeError):
            self.guard.launch()
        self.torch.cuda.synchronize.side_effect = RuntimeError("device failure")
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            self.guard.finalize()
        self.assertTrue(self.guard.inflight)
        self.assertFalse(self.guard.closed)
        self.lease.close.assert_not_called()
        self.kernel.finalize.assert_not_called()

    def test_changed_implementation_rejected_before_launch(self):
        self.pk._paged_gqa_contract["implementation"] = "portable"
        with self.assertRaisesRegex(RuntimeError, "implementation changed"):
            self.guard.launch()
        self.kernel.launch.assert_not_called()
        self.assertFalse(self.guard.inflight)


class MetadataTests(unittest.TestCase):
    def test_page_boundaries_and_empty_requests(self):
        pages = [[2*i, 2*i+1] for i in range(8)]
        contract.validate_metadata(pages, [0, 1, 63, 64, 65, 127, 128, 0], [0]*8, 16)
        for length in (-1, 129):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "capacity"):
                contract.validate_metadata(pages, [length]+[0]*7, [0]*8, 16)

    def test_missing_page_allowed_only_outside_visible_window(self):
        pages = [[-1, 2*i+1] for i in range(8)]
        # At length 65, a one-key window reads only position 64 on page 1.
        contract.validate_metadata(pages, [65]*8, [1]*8, 16)
        for window in (0, 2):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "visible page missing"):
                contract.validate_metadata(pages, [65]*8, [window]*8, 16)

    def test_shared_invisible_pages_still_violate_ownership(self):
        pages = [[0, i+1] for i in range(8)]
        with self.assertRaisesRegex(ValueError, "exclusive page ownership"):
            contract.validate_metadata(pages, [65]*8, [1]*8, 9)

    def test_invalid_page_indices_rejected_even_when_request_empty(self):
        for page in (-2, 8):
            with self.subTest(page=page), self.assertRaisesRegex(ValueError, "outside allocation"):
                contract.validate_metadata([[page]]+[[-1]]*7, [0]*8, [0]*8, 8)


class LeaseTests(unittest.TestCase):
    def lease(self, writes=(), reads=()):
        lease = bindings.CacheLease(writes, reads)
        self.addCleanup(lease.close)
        return lease

    def test_readers_coexist_but_exclude_writers_until_all_close(self):
        region = (0, 4096, 8192)
        first = self.lease(reads=[region])
        second = self.lease(reads=[region])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.lease(writes=[region])
        first.close()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.lease(writes=[region])
        second.close()
        self.lease(writes=[region])

    def test_partial_write_overlap_rejected_but_adjacent_or_other_device_allowed(self):
        self.lease(writes=[(0, 4096, 8192)])
        for region in ((0, 4095, 4097), (0, 8191, 8193)):
            with self.subTest(region=region), self.assertRaisesRegex(ValueError, "conflicts"):
                self.lease(reads=[region])
        self.lease(writes=[(0, 8192, 9000), (1, 4096, 8192)])


class DescriptorTests(unittest.TestCase):
    def specs(self):
        shapes = {"query": (8, 8, 8, 128), "keys": (72, 64, 8, 128),
                  "values": (72, 64, 8, 128), "pages": (8, 9),
                  "lengths": (8,), "windows": (8,), "new_keys": (8, 8, 128),
                  "new_values": (8, 8, 128), "output": (8, 8192)}
        return {name: dict(shape=shape, strides=contract.contiguous_strides(shape),
                          dtype="int32" if name in ("pages", "lengths", "windows") else "bfloat16")
                for name, shape in shapes.items()}

    def test_valid_partition_count_and_bad_descriptors(self):
        self.assertEqual(contract.validate_specs(self.specs(), 128)["chunks"], 5)
        for role, field, value in (("query", "dtype", "float32"),
                                   ("output", "shape", (8, 4096)),
                                   ("pages", "strides", (10, 1))):
            specs = self.specs()
            specs[role][field] = value
            with self.subTest(role=role, field=field), self.assertRaises(ValueError):
                contract.validate_specs(specs, 128)
        for chunk in (0, 63, 65, 32769, True):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                contract.validate_specs(self.specs(), chunk)

    def test_base_alignment_and_writable_aliases(self):
        for dtype, alignment in (("bfloat16", 16), ("float32", 16), ("int32", 4)):
            bindings.validate_alignment(4096, dtype)
            for address in (0, 4097, 4096 + alignment - 1):
                with self.subTest(dtype=dtype, address=address), self.assertRaises(ValueError):
                    bindings.validate_alignment(address, dtype)
        with self.assertRaisesRegex(ValueError, "aliases"):
            bindings.validate_ranges({"cache": (0, 4096, 8192),
                                      "query": (0, 8191, 9000)}, ["cache"])
        bindings.validate_ranges({"cache": (0, 4096, 8192),
                                  "query": (0, 8192, 9000)}, ["cache"])

    def test_changed_pointer_rejected_before_launch(self):
        tensor = SimpleNamespace(data_ptr=lambda: 4096, shape=(8,), stride=lambda: (1,),
                                 dtype="int32", device="cuda:0", untyped_storage=lambda: object())
        kernel = Mock()
        guard = bindings.GuardedGQA(kernel, SimpleNamespace(_paged_gqa_contract={}),
                                    {"lengths": tensor}, Mock())
        tensor.data_ptr = lambda: 8192
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            guard.launch()
        kernel.launch.assert_not_called()


class MetadataSnapshotOrderingTests(unittest.TestCase):
    def test_sync_finishes_before_any_metadata_copy(self):
        order = []
        class Tensor:
            def __init__(self, name): self.name = name
            def detach(self): order.append(('detach', self.name)); return self
            def cpu(self): order.append(('cpu', self.name)); return self
            def tolist(self): order.append(('tolist', self.name)); return [self.name]
        names = {name: name for name in ('pages', 'lengths', 'windows')}
        tensors = {name: Tensor(name) for name in names}
        from contextlib import nullcontext
        torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda dev: nullcontext(), is_current_stream_capturing=lambda: False, synchronize=lambda dev: order.append(('sync', dev))))
        with patch.dict('sys.modules', {'torch': torch}):
            result = bindings._metadata_snapshot(tensors, names)
        self.assertEqual(order[0], ('sync', 0))
        self.assertEqual([item for item in order if item[0] == 'cpu'],
                         [('cpu', 'pages'), ('cpu', 'lengths'), ('cpu', 'windows')])
        self.assertEqual(result, (['pages'], ['lengths'], ['windows']))

    def test_active_capture_rejected_before_sync_or_metadata_copy(self):
        tensor = Mock()
        from contextlib import nullcontext
        cuda = SimpleNamespace(device=lambda dev: nullcontext(), is_current_stream_capturing=Mock(return_value=True), synchronize=Mock())
        with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(RuntimeError, 'outside CUDA graph capture'):
                bindings._metadata_snapshot({x: tensor for x in ('pages', 'lengths', 'windows')},
                                            {x: x for x in ('pages', 'lengths', 'windows')})
        cuda.synchronize.assert_not_called()
        tensor.detach.assert_not_called()

    def test_failed_sync_prevents_metadata_read(self):
        tensor = Mock()
        from contextlib import nullcontext
        torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda dev: nullcontext(), is_current_stream_capturing=lambda: False, synchronize=Mock(side_effect=RuntimeError('sync failed'))))
        with patch.dict('sys.modules', {'torch': torch}):
            with self.assertRaisesRegex(RuntimeError, 'sync failed'):
                bindings._metadata_snapshot({x: tensor for x in ('pages', 'lengths', 'windows')},
                                            {x: x for x in ('pages', 'lengths', 'windows')})
        tensor.detach.assert_not_called()


class StaticSM100ContractDocumentationTests(unittest.TestCase):
    def test_sm100_shape_and_error_describe_d128(self):
        import ast
        tree = ast.parse((ROOT / "python/mirage/mpk/static_megakernel.py").read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "paged_gqa_decode_layer")
        doc = ast.get_docstring(method)
        errors = [node.value for node in ast.walk(method)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertIn("query [8,8,8,128]", doc)
        self.assertTrue(any("experimental sm100 requires G8/H8/D128" in msg for msg in errors))
        self.assertTrue(any("only partitioned sm100 GQA" in msg for msg in errors))
        self.assertIn("finish and order prior writes to pages, lengths, and windows", doc)
        self.assertIn("prevent concurrent writes from validation", doc)


if __name__ == "__main__":
    unittest.main()
