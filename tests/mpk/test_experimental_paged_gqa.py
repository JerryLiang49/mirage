"""Opt-in integration tests for guarded cooperative paged decode on SM100.

Run in an isolated process with a timeout; cooperative scheduling can fail
under shared-device contention. This suite performs four native compilations.
"""
import itertools
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('MIRAGE_RUN_EXPERIMENTAL_GQA') != '1',
    reason='set MIRAGE_RUN_EXPERIMENTAL_GQA=1 for expensive SM100 GPU tests')


@pytest.mark.parametrize('partition,with_sinks,with_projection', [
    (4160, True, False), (4160, False, False), (8256, True, False),
    (4160, True, True),
])
def test_cooperative_paged_gqa_boundaries(tmp_path, partition, with_sinks, with_projection):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        pytest.skip('requires exactly one visible CUDA device')
    if torch.cuda.get_device_capability(0) != (10, 0):
        pytest.skip('requires SM100')
    from mirage.mpk.static_megakernel import StaticMegakernel
    from mirage.mpk import compiler
    import gqa_reference

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    fixture = gqa_reference.make_fixture(with_sinks=with_sinks)
    tensors, backings = {}, {}
    inputs = dict(fixture)
    if with_projection:
        generator = torch.Generator(device='cpu').manual_seed(2947)
        inputs['projection_weight'] = (
            torch.randn((128, 8192), generator=generator) * 0.125).to(torch.bfloat16)
        inputs['projection_output'] = torch.full((8, 128), float('nan'), dtype=torch.float32)
    for name, cpu in inputs.items():
        size = cpu.numel() * cpu.element_size()
        back = torch.full((size + 512,), 90, dtype=torch.uint8, device='cuda')
        tensor = back[256:-256].view(cpu.dtype).reshape(cpu.shape)
        tensor.copy_(cpu)
        tensors[name], backings[name] = tensor, back
    torch.cuda.synchronize()

    pk = StaticMegakernel(num_gpus=1, num_sms=128)
    descriptors = {name: pk.attach_input(tensors[name], name=name) for name in fixture}
    pk.paged_gqa_decode_layer(**descriptors, partition_tokens=partition)
    if with_projection:
        weight = pk.attach_input(tensors['projection_weight'], name='projection_weight')
        projected = pk.attach_input(tensors['projection_output'], name='projection_output')
        pk.gemm_tile_layer(descriptors['output'], weight, projected, combine='store')
    nodes = compiler.graph_nodes(pk)
    producer, merge = nodes[:2]
    queues = [[] for _ in range(128)]
    # Manual producer-first placement; no measured-cost search is used.
    for i, pos in enumerate(itertools.product(*(range(n) for n in producer.grid))):
        queues[(73 * i + 19) % 128].append((producer.graph_idx, pos))
    for i, pos in enumerate(itertools.product(*(range(n) for n in merge.grid))):
        queues[(37 * i + 11) % 128].append((merge.graph_idx, pos))
    if with_projection:
        projection = nodes[2]
        assert projection.grid == (1, 1, 1)
        queues[0].append((projection.graph_idx, (0, 0, 0)))
    paths, _ = compiler.compile_plan(pk, {'grids': {}, 'lists': [queues]}, str(tmp_path))
    kernel = pk.compile_static(paths, gpu_tensors=[tensors], out_dir=str(tmp_path),
                               cooperative_gqa=True)
    try:
        with pytest.raises(RuntimeError, match='cooperative'):
            kernel.capture_graph()
        saved_length = int(tensors['lengths'][0].item())
        tensors['lengths'][0] = 8257
        with pytest.raises(ValueError, match='length outside page capacity'):
            kernel.launch()
        tensors['lengths'][0] = saved_length
        for _ in range(2):
            kernel.launch()
            kernel.wait()
            observed = {name: tensor.cpu() for name, tensor in tensors.items()}
            guards = {name: torch.cat((back[:256], back[-256:])).cpu()
                      for name, back in backings.items()}
            gqa_reference.check(fixture, {name: observed[name] for name in fixture},
                                {name: guards[name] for name in fixture})
            if with_projection:
                assert torch.equal(observed['projection_weight'], inputs['projection_weight'])
                reference = observed['output'].double() @ inputs['projection_weight'].double().T
                torch.testing.assert_close(observed['projection_output'].double(), reference,
                                           rtol=1e-3, atol=1e-3)
                for name in ('projection_weight', 'projection_output'):
                    assert bool((guards[name] == 90).all()), 'projection red-zone corruption'
            counters = torch.frombuffer(bytearray(kernel.read(0, 'counters')),
                                        dtype=torch.int32).clone()
            expected = torch.zeros_like(counters)
            expected[producer.info['counter']:producer.info['counter'] + 8] = (
                8 * pk._paged_gqa_contract['chunks'])
            expected[merge.info['counter']] = 64
            if with_projection:
                expected[projection.info['counter']] = 1
            assert torch.equal(counters, expected), 'task counters did not reset or complete'
    finally:
        kernel.finalize()
