"""Experimental correctness wrapper; checks synchronize CPU metadata.
Generic lease/range helpers reused from the earlier bounded static-attention
prototype. Managed exclusion covers this process/wrapper only.
"""
import threading
_LOCK=threading.Lock()
_LEASES={}
def overlap(a, b):
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]

def validate_ranges(ranges, writable_names):
    """Reject every writable/read-write alias; readonly-readonly views may alias.

    Ranges are device-index, start-address, end-address of actual tensor bytes.
    """
    for name in writable_names:
        a = ranges[name]
        if a[1] <= 0 or a[2] <= a[1]:
            raise ValueError('invalid writable pointer range')
        for other, b in ranges.items():
            if other != name and overlap(a, b):
                raise ValueError(f'writable {name} aliases bound tensor {other}')

class CacheLease:
    """Managed read/write footprints: writes exclude other reads and writes."""
    def __init__(self, ranges, readonly_ranges=()):
        self.token = object()
        self.closed = False
        with _LOCK:
            writers, readers = tuple(ranges), tuple(readonly_ranges)
            for old_writers, old_readers in _LEASES.values():
                if (any(overlap(a,b) for a in writers for b in old_writers + old_readers)
                        or any(overlap(a,b) for a in readers for b in old_writers)):
                    raise ValueError('storage conflicts with a managed static kernel lease')
            _LEASES[self.token] = (writers, readers)

    def close(self):
        with _LOCK:
            _LEASES.pop(self.token, None)
            self.closed = True

def binding_signature(tensors):
    return {name: (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype),
                   str(t.device)) for name, t in tensors.items()}

def bound_ranges(tensors):
    return {name: (t.device.index, t.data_ptr(),
                   t.data_ptr() + t.numel() * t.element_size())
            for name, t in tensors.items()}

def validate_alignment(pointer, dtype):
    """Conservative base alignment for the supported vector/TMA binding scope.

    Contiguity alone does not guarantee an aligned base for a sliced tensor.
    """
    if dtype in ('bfloat16', 'float32'):
        alignment = 16
    elif dtype == 'int32':
        alignment = 4
    else:
        raise ValueError('unsupported bound tensor dtype')
    if type(pointer) is not int or pointer <= 0 or pointer % alignment:
        raise ValueError(f'{dtype} binding requires {alignment}-byte base alignment')

def projection_bindings(pk, nodes=None):
 """Permit only one unpolled store-mode projection after the GQA merge.
 Extra descriptors are derived from the actual graph, never from caller keys.
 """
 from .compiler import graph_nodes
 from ..core import bfloat16, float32
 nodes = graph_nodes(pk) if nodes is None else nodes
 record = pk._paged_gqa_contract
 if record.get('mode') == 'direct' or record.get('implementation') != 'sm100':
  raise ValueError('only partitioned sm100 GQA is supported')
 expected = {record['producer'], record['merge']}
 registered = [n for n in nodes if n.name in ('gqa_paged_parts', 'gqa_merge_ready', 'gqa_direct')]
 if {n.graph_idx for n in registered} != expected or len(registered) != 2:
  raise ValueError('GQA nodes differ from declared ownership contract')
 extra = [n for n in nodes if n.graph_idx not in expected]
 if not extra:
  return {}, []
 if len(extra) != 1:
  raise ValueError('only one dependent projection is supported')
 node = extra[0]
 if node.name != 'gemm_tile' or len(node.inputs) != 2 or len(node.outputs) != 1:
  raise ValueError('only one dependent GEMM projection is supported')
 act, weight = node.inputs
 output = node.outputs[0]
 def shape(t): return tuple(t.dim(i) for i in range(t.num_dims))
 width = record['groups'] * record['heads'] * record['dim']
 projection_shape=shape(weight)
 rows=projection_shape[0] if len(projection_shape)==2 else 0
 tile_rows=128 if rows%128==0 else 64
 if (act.guid != record['tensor_guids']['output'] or
     shape(act) != (8,width) or rows<=0 or rows%64 or shape(weight) != (rows,width) or
     shape(output) != (8,rows) or weight.dtype != bfloat16 or output.dtype != float32 or
     node.params != [2,width,rows,0] or tuple(node.grid) != (rows//tile_rows,1,1) or
     node.info['rows_split_over_gpus'] or node.info['exchange']):
  raise ValueError('unsupported projection layout or mode')
 protected = set(record['tensor_guids'].values())
 if weight.guid in protected or output.guid in protected or weight.guid == output.guid:
  raise ValueError('projection descriptors must be distinct from GQA bindings')
 specs = {}
 for t, dtype in ((weight,'bfloat16'),(output,'float32')):
  name = pk._tensor_names[t.guid]
  if name in record['names'].values() or name in specs:
   raise ValueError('projection binding names must be distinct')
  specs[name] = dict(shape=shape(t),strides=tuple(t.stride),dtype=dtype)
 return specs, [pk._tensor_names[output.guid]]

def _metadata_snapshot(tensors, names):
 # Wait for work already queued in this CUDA context before reading metadata.
 # External or subsequently enqueued mutation remains a caller responsibility.
 import torch
 with torch.cuda.device(0):
  if torch.cuda.is_current_stream_capturing():
   raise RuntimeError('GQA metadata validation must run outside CUDA graph capture')
  torch.cuda.synchronize(0)
  return tuple(tensors[names[r]].detach().cpu().tolist()
               for r in ('pages', 'lengths', 'windows'))

def validate_bound(pk,tensors):
 from .paged_gqa_contract import validate_specs,validate_metadata
 record=pk._paged_gqa_contract
 names=record['names']
 extras,extra_writers=projection_bindings(pk)
 if set(tensors)!=set(names.values())|set(extras):
  raise ValueError('experimental GQA binding requires exactly its declared tensors')
 specs={}
 for role,name in names.items():
  tensor=tensors[name]
  if tensor.device.type!='cuda' or tensor.device.index!=0:
   raise ValueError('GQA bindings must reside on logical CUDA device0')
  specs[role]=dict(shape=tuple(tensor.shape),strides=tuple(tensor.stride()),dtype=str(tensor.dtype).removeprefix('torch.'))
  if specs[role]!=record['specs'][role]:
   raise ValueError(f'{role}: binding descriptor differs from graph declaration')
  validate_alignment(tensor.data_ptr(),specs[role]['dtype'])
 checked=validate_specs(specs,record['chunk_size'])
 if checked['chunks']!=record['chunks']:
  raise ValueError('partition count differs from the graph contract')
 from .compiler import graph_nodes
 nodes={n.graph_idx:n for n in graph_nodes(pk)}
 producer=nodes[record['producer']]
 if record.get('mode') == 'direct' or record.get('implementation') != 'sm100':
  raise ValueError('only partitioned sm100 GQA is supported')
 merge=nodes[record['merge']]
 expected=[checked[k] for k in ['groups','heads','dim','chunks','chunk_size','page_size','page_capacity']]
 if tuple(checked[k] for k in ('groups','heads','dim','page_capacity'))!=(8,8,128,129) or checked['chunk_size'] not in (4160,8256):
  raise ValueError('unsupported sm100 declaration')
 if (producer.name!='gqa_paged_parts' or merge.name!='gqa_merge_ready' or
     producer.params!=expected or merge.params!=expected[:4]+[int(checked['sinks'])] or
     tuple(producer.grid)!=(checked['groups'],checked['chunks'],8) or
     tuple(merge.grid)!=(checked['groups'],1,8)):
  raise ValueError('task parameters or grid differ from the implementation contract')
 ranges=bound_ranges(tensors)
 for name,spec in extras.items():
  t=tensors[name]
  if (t.device.type!='cuda' or t.device.index!=0 or
      dict(shape=tuple(t.shape),strides=tuple(t.stride()),dtype=str(t.dtype).removeprefix('torch.'))!=spec):
   raise ValueError('projection binding descriptor differs from declaration')
  validate_alignment(t.data_ptr(),spec['dtype'])
 writers=[names[r] for r in ['keys','values','output']]+extra_writers
 validate_ranges(ranges,writers)
 pages,lengths,windows=_metadata_snapshot(tensors,names)
 validate_metadata(pages,lengths,windows,record['physical_pages'])
 return ranges,[ranges[n] for n in writers],[v for n,v in ranges.items() if n not in writers]

class GuardedGQA:
 """Checked correctness path. Validation runs outside CUDA graph capture."""
 def __init__(self,kernel,pk,tensors,lease,cooperative=False):
  self.cooperative=cooperative
  self._kernel=kernel;self.pk=pk;self.tensors=dict(tensors);self.lease=lease
  self.implementation=pk._paged_gqa_contract.get('implementation','sm100')
  self.signature=binding_signature(self.tensors)
  self.storages=[t.untyped_storage() for t in self.tensors.values()]
  self.lock=threading.Lock();self.inflight=False;self.closed=False
  self._graph_completion=None;self._sync_all=False
 def __getattr__(self,name):
  if name in ('layout','elapsed_us','read','num_gpus','cu_path','so_path'):
   return getattr(self._kernel,name)
  raise AttributeError(f'GuardedGQA does not expose {name}')
 def launch_on_stream(self, stream):
  """External stream launch is unsupported by the checked ownership wrapper.

  Use launch()/wait(), or capture_graph() and its checked replay wrapper.
  The underlying stream entrypoint is reserved for capture_graph().
  """
  raise RuntimeError("GuardedGQA does not support external launch_on_stream; use launch or capture_graph")
 def launch_stream(self, *args, **kwargs):
  raise RuntimeError("GuardedGQA does not support external launch_stream; use launch or capture_graph")
 def rebind(self, *args, **kwargs):
  raise RuntimeError("GuardedGQA does not support rebind; compile with new bindings")
 def _check_idle(self):
  if self.closed:raise RuntimeError('GQA kernel finalized')
  if self.inflight:raise RuntimeError('wait/done required before GQA relaunch')
  if binding_signature(self.tensors)!=self.signature:raise RuntimeError('GQA binding changed')
  if self.pk._paged_gqa_contract.get('implementation','sm100')!=self.implementation:raise RuntimeError('GQA implementation changed')
  validate_bound(self.pk,self.tensors)
 def launch(self,flush_bytes=0):
  with self.lock:
   self._check_idle()
   import torch
   torch.cuda.synchronize(0)
   self.inflight=True;self._sync_all=True
   # A failed launch may enqueue work without recording its completion event.
   # Keep device-wide synchronization as the recovery path until launch returns.
   self._kernel.launch(flush_bytes)
   self._sync_all=False
 def _wait_locked(self):
  if not self.inflight:return
  if self._sync_all:
   import torch
   torch.cuda.synchronize(0)
  elif self._graph_completion is not None:self._graph_completion.synchronize()
  else:self._kernel.wait()
  self.inflight=False;self._graph_completion=None;self._sync_all=False
 def wait(self):
  with self.lock:
   if self.closed:raise RuntimeError('GQA kernel finalized')
   self._wait_locked()
 def done(self):
  with self.lock:
   if self.closed:raise RuntimeError('GQA kernel finalized')
   if not self.inflight:return True
   if self._sync_all:return False
   done=self._graph_completion.query() if self._graph_completion is not None else self._kernel.done()
   if done:self.inflight=False;self._graph_completion=None
   return done
 def capture_graph(self):
  """Capture a serving layer after preflight, retaining its managed ownership.
  The returned replay wrapper still validates metadata outside the graph.
  This is not an enclosing serving-engine capture interface or a timed path.
  """
  if self.cooperative:raise RuntimeError('cooperative GQA does not support graph capture')
  with self.lock:
   self._check_idle()
   if not self._kernel.layout()['serving']:
    raise RuntimeError('graph capture requires serving-mode device resets')
   import torch
   with torch.cuda.device(0):
    if torch.cuda.is_current_stream_capturing():
     raise RuntimeError('capture_graph must be called outside existing capture')
    torch.cuda.synchronize(0)
    # Until successful synchronization, failures retain conservative ownership.
    self.inflight=True;self._sync_all=True
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):self._kernel.launch_on_stream(stream)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):self._kernel.launch_on_stream(stream)
    torch.cuda.synchronize(0)
    self.inflight=False;self._sync_all=False
    return CapturedGQA(self,graph)
 def finalize(self):
  with self.lock:
   if not self.closed:
    if self.inflight:self._wait_locked()
    self._kernel.finalize();self.lease.close();self.closed=True

class CapturedGQA:
 """Owns a captured graph and retains its parent/buffers/lease by reference."""
 def __init__(self,parent,graph):self.parent=parent;self.graph=graph
 def replay(self):
  parent=self.parent
  with parent.lock:
   parent._check_idle()
   import torch
   with torch.cuda.device(0):
    torch.cuda.synchronize(0)
    event=torch.cuda.Event()
    parent.inflight=True;parent._sync_all=True
    self.graph.replay()
    # The generated launcher's events were captured, so do not use them as
    # host completion handles. Record a fresh event after this graph replay.
    event.record(torch.cuda.current_stream())
    parent._graph_completion=event;parent._sync_all=False
 def wait(self):self.parent.wait()
 def done(self):return self.parent.done()
