"""Descriptor and metadata checks for the experimental static paged GQA path."""
import math

def contiguous_strides(shape):
 result=[];stride=1
 for d in reversed(shape):result.append(stride);stride*=d
 return tuple(reversed(result))

def validate_specs(specs,partition_tokens=128):
 if type(partition_tokens) is not int or not 64<=partition_tokens<=32768 or partition_tokens%64:
  raise ValueError('partition_tokens must be an integer multiple of64 in[64,32768]')
 required={'query','keys','values','pages','lengths','windows','new_keys','new_values','output'}
 if set(specs) not in [required,required|{'sinks'}]:
  raise ValueError('paged GQA requires query, cache, page metadata, new KV and output')
 for role,s in specs.items():
  shape=tuple(s['shape'])
  if not shape or any(type(d)!=int or d<=0 for d in shape):
   raise ValueError(f'{role}: positive dimensions required')
  if tuple(s['strides'])!=contiguous_strides(shape):
   raise ValueError(f'{role}: contiguous layout required')
  wanted='int32' if role in {'pages','lengths','windows'} else 'bfloat16'
  if s['dtype']!=wanted:raise ValueError(f'{role}: expected {wanted}')
 q=tuple(specs['query']['shape'])
 if len(q)!=4 or q[0]!=8 or q[2]!=8 or q[3] not in [64,128]:
  raise ValueError('query must be [8, KVheads, 8, D64/128]')
 t,g,h,d=q
 shapes={r:tuple(v['shape']) for r,v in specs.items()}
 cache=shapes['keys'];pages=shapes['pages']
 if len(cache)!=4 or cache[1:]!=(64,g,d) or shapes['values']!=cache:
  raise ValueError('cache must be matching [physical_pages,64,KVheads,D]')
 if len(pages)!=2 or pages[0]!=t or pages[1]*64>32768:
  raise ValueError('pages must be [8, logical_pages], capacity<=32768')
 expected={'output':(t,g*h*d),'new_keys':(t,g,d),'new_values':(t,g,d),'lengths':(t,),'windows':(t,)}
 if 'sinks' in specs:expected['sinks']=(g,h)
 for role,want in expected.items():
  if shapes[role]!=want:raise ValueError(f'{role}: expected {want}')
 return dict(tokens=t,groups=g,heads=h,dim=d,page_size=64,page_capacity=pages[1],physical_pages=cache[0],chunk_size=partition_tokens,chunks=(pages[1]*64+partition_tokens-1)//partition_tokens,sinks='sinks' in specs)

def validate_metadata(pages,lengths,windows,physical_pages):
 """CPU metadata values only. Graph replay callers must preserve this contract."""
 if len(pages)!=8 or len(lengths)!=8 or len(windows)!=8:
  raise ValueError('eight independent requests required')
 width=len(pages[0]);seen=set()
 if width<1 or any(len(row)!=width for row in pages):raise ValueError('rectangular page table required')
 for row,n,w in zip(pages,lengths,windows):
  if type(n)!=int or not 0<=n<=width*64:raise ValueError('length outside page capacity')
  if type(w)!=int or w<0:raise ValueError('window must be nonnegative')
  lo=max(0,n-w) if w else 0
  for logical,page in enumerate(row):
   if type(page)!=int or page < -1 or page>=physical_pages:raise ValueError('page outside allocation')
   if page>=0:
    if page in seen:raise ValueError('shared physical pages unsupported; exclusive page ownership required')
    seen.add(page)
   if logical*64<n and (logical+1)*64>lo and page<0:
    raise ValueError('visible page missing')


def validate_device(record, capability):
    """The sole supported partitioned implementation requires SM100."""
    if record.get("implementation") != "sm100" or record.get("mode") == "direct":
        raise ValueError("only partitioned sm100 GQA is supported")
    if tuple(capability) != (10, 0):
        raise ValueError("experimental sm100 GQA requires SM100")
