"""Opt-in SM100 regression for finite logits around the old raw-dot sentinel.

Run with the same isolated-process timeout as test_experimental_paged_gqa.py.
This test compiles one cooperative kernel and checks two stationary launches.
It does not establish correctness for overflowing FP32 intermediates.
"""
import itertools
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('MIRAGE_RUN_EXPERIMENTAL_GQA') != '1',
    reason='set MIRAGE_RUN_EXPERIMENTAL_GQA=1 for expensive SM100 GPU tests')


def test_cooperative_gqa_finite_negative_logits(tmp_path):
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
    query = torch.zeros((8, 8, 8, 128), dtype=torch.bfloat16)
    query[..., :2] = 1
    keys = torch.zeros((1032, 64, 8, 128), dtype=torch.bfloat16)
    new_keys = torch.zeros((8, 8, 128), dtype=torch.bfloat16)
    # BF16 pairs (-49920, -80)/(-49920, -79) give exact -50000/-49999 dots.
    # A scalar BF16 conversion of those integers would lose the distinction.
    key_pairs = ((0, 0), (-65536, 0), (-50176, 0), (-49920, -80),
                 (-49920, -79), (0, 0), (-65536, 0), (-49920, -79))
    for request, (first, second) in enumerate(key_pairs):
        keys[request * 129:(request + 1) * 129, ..., 0] = first
        keys[request * 129:(request + 1) * 129, ..., 1] = second
        new_keys[request, :, 0] = first
        new_keys[request, :, 1] = second
    fixture = {
        'query': query, 'keys': keys, 'values': torch.full_like(keys, 1.25),
        'pages': torch.arange(1032, dtype=torch.int32).reshape(8, 129),
        'lengths': torch.tensor((0, 1, 15, 16, 17, 63, 64, 65), dtype=torch.int32),
        'windows': torch.tensor((0, 0, 0, 0, 1, 16, 63, 64), dtype=torch.int32),
        'new_keys': new_keys, 'new_values': torch.full_like(new_keys, 1.25),
        'output': torch.full((8, 8192), float('nan'), dtype=torch.bfloat16),
    }
    tensors, backings = {}, {}
    for name, cpu in fixture.items():
        backing = torch.full((cpu.numel() * cpu.element_size() + 512,),
                             90, dtype=torch.uint8, device='cuda')
        tensor = backing[256:-256].view(cpu.dtype).reshape(cpu.shape)
        tensor.copy_(cpu)
        tensors[name], backings[name] = tensor, backing
    torch.cuda.synchronize()
    pk = StaticMegakernel(num_gpus=1, num_sms=128)
    descriptors = {name: pk.attach_input(tensor, name=name)
                   for name, tensor in tensors.items()}
    pk.paged_gqa_decode_layer(**descriptors, partition_tokens=4160)
    producer, merge = compiler.graph_nodes(pk)
    queues = [[] for _ in range(128)]
    for i, position in enumerate(itertools.product(*(range(n) for n in producer.grid))):
        queues[(73 * i + 19) % 128].append((producer.graph_idx, position))
    for i, position in enumerate(itertools.product(*(range(n) for n in merge.grid))):
        queues[(37 * i + 11) % 128].append((merge.graph_idx, position))
    paths, _ = compiler.compile_plan(pk, {'grids': {}, 'lists': [queues]}, str(tmp_path))
    kernel = pk.compile_static(paths, gpu_tensors=[tensors], out_dir=str(tmp_path),
                               cooperative_gqa=True)
    try:
        for _ in range(2):
            kernel.launch()
            kernel.wait()
            observed = {name: tensor.cpu() for name, tensor in tensors.items()}
            guards = {name: torch.cat((backing[:256], backing[-256:])).cpu()
                      for name, backing in backings.items()}
            gqa_reference.check(fixture, observed, guards)
            for name in set(fixture) - {'keys', 'values', 'output'}:
                assert torch.equal(observed[name].contiguous().view(torch.uint8),
                                   fixture[name].contiguous().view(torch.uint8))
            # Uniform values provide a semantic control independent of logit scale.
            assert bool((observed['output'][0] == 0).all())
            assert bool((observed['output'][1:] == 1.25).all())
            counters = torch.frombuffer(bytearray(kernel.read(0, 'counters')),
                                        dtype=torch.int32).clone()
            assert counters.shape == (512,)
            expected = torch.zeros_like(counters)
            expected[producer.info['counter']:producer.info['counter'] + 8] = 16
            expected[merge.info['counter']] = 64
            assert torch.equal(counters, expected)
    finally:
        kernel.finalize()
