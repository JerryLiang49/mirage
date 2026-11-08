"""Opt-in SM100 cooperative GQA advancing and dynamic-page correctness regression."""
import itertools,os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('MIRAGE_RUN_EXPERIMENTAL_GQA')!='1',
                              reason='set MIRAGE_RUN_EXPERIMENTAL_GQA=1 for GPU validation')

@pytest.mark.parametrize('scenario', ['fixed_pages', 'dynamic_pages', 'split_crossing'])
def test_advancing_paged_gqa(tmp_path, scenario):
 dynamic_pages=scenario=='dynamic_pages'
 torch=pytest.importorskip('torch')
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:pytest.skip('one visible CUDA device required')
 if torch.cuda.get_device_capability(0)!=(10,0):pytest.skip('SM100 required')
 from mirage.mpk import compiler
 from mirage.mpk.static_megakernel import StaticMegakernel
 import gqa_reference as ref
 torch.set_num_threads(1);torch.cuda.set_device(0)
 if dynamic_pages:
  from gqa_trajectory import Planner,INITIAL,WINDOWS
  steps=16;initial=list(INITIAL);windows=list(WINDOWS);planner=Planner(seed=5527)
  f=ref.make_fixture(seed=5527,with_sinks=True)
  f['pages']=planner.step(initial,windows);step_seed=5551
 else:
  steps=12;initial=[60,61,62,63,64,4155,4159,8245];windows=[0,1,63,64,65,64,129,0]
  if scenario=='split_crossing':
   steps=3;initial=[127]*8;windows=[0,1,63,64,65,128,129,0]
  f=ref.make_fixture(seed=1947,with_sinks=True);step_seed=9103
  gen=torch.Generator().manual_seed(5237);perm=torch.randperm(1032,generator=gen);pages=torch.full((8,129),-1,dtype=torch.int32);used=0
  for r,n in enumerate(initial):
   count=(min(8256,n+11)+63)//64;pages[r,:count]=perm[used:used+count];used+=count
  assert used<=1032 and len(set(perm[:used].tolist()))==used
  f['pages']=pages
 f['lengths']=torch.tensor(initial,dtype=torch.int32);f['windows']=torch.tensor(windows,dtype=torch.int32)
 tensors={};backs={}
 for name,cpu in f.items():
  size=cpu.numel()*cpu.element_size();back=torch.full((size+512,),90,dtype=torch.uint8,device='cuda')
  tensor=back[256:-256].view(cpu.dtype).reshape(cpu.shape);tensor.copy_(cpu);tensors[name]=tensor;backs[name]=back
 torch.cuda.synchronize()
 pk=StaticMegakernel(num_gpus=1,num_sms=128);desc={n:pk.attach_input(t,name=n) for n,t in tensors.items()};pk.paged_gqa_decode_layer(**desc,partition_tokens=4160)
 producer,merge=compiler.graph_nodes(pk);queues=[[] for _ in range(128)]
 for i,pos in enumerate(itertools.product(*(range(n) for n in producer.grid))):queues[(73*i+19)%128].append((producer.graph_idx,pos))
 for i,pos in enumerate(itertools.product(*(range(n) for n in merge.grid))):queues[(37*i+11)%128].append((merge.graph_idx,pos))
 paths,_=compiler.compile_plan(pk,{'grids':{},'lists':[queues]},str(tmp_path));kernel=pk.compile_static(paths,gpu_tensors=[tensors],out_dir=str(tmp_path),cooperative_gqa=True)
 rng=torch.Generator().manual_seed(step_seed)
 try:
  for step in range(steps):
   f['lengths']=torch.tensor([n+step for n in initial],dtype=torch.int32)
   if dynamic_pages:f['pages']=planner.step(f['lengths'].tolist(),windows)
   for name in ('query','new_keys','new_values','sinks'):f[name]=(torch.randn(f[name].shape,generator=rng)*0.5).to(torch.bfloat16)
   for name in ('lengths','pages','query','new_keys','new_values','sinks'):tensors[name].copy_(f[name])
   torch.cuda.synchronize();kernel.launch();kernel.wait()
   obs={name:t.cpu() for name,t in tensors.items()};guards={name:torch.cat((b[:256],b[-256:])).cpu() for name,b in backs.items()}
   counters=torch.frombuffer(bytearray(kernel.read(0,'counters')),dtype=torch.int32).clone();expected=torch.zeros_like(counters)
   expected[producer.info['counter']:producer.info['counter']+8]=8*pk._paged_gqa_contract['chunks'];expected[merge.info['counter']]=64
   assert torch.equal(counters,expected),f'counters step {step}'
   ref.check(f,obs,guards);updated=ref.reference(f);f['keys']=updated['keys'];f['values']=updated['values']
  if dynamic_pages:assert planner.releases>0 and planner.reuses>0
 finally:kernel.finalize()
