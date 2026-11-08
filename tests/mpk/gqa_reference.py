"""Independent CPU-only reference and strict gate for one BF16 paged GQA step.

lengths include the appended token. windows=0 selects full history; positive
windows select the last W tokens. A finite sink is a natural-logit zero-value
entry. Page sharing is forbidden. No production code is imported.
"""
import math
import torch

LENGTHS=(0,1,63,64,65,4160,4161,8256)
WINDOWS=(0,0,2,64,1,65,64,0)
ABS_MAX=0.05
RRMS_MAX=0.01


def make_fixture(seed=1947, with_sinks=True):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    def rand(shape):
        return (torch.randn(shape,generator=gen,dtype=torch.float32)*0.5).to(torch.bfloat16)
    pages=torch.full((8,129),-1,dtype=torch.int32)
    lengths=torch.tensor(LENGTHS,dtype=torch.int32)
    windows=torch.tensor(WINDOWS,dtype=torch.int32)
    perm=torch.randperm(1032,generator=gen)
    next_page=0
    # Assign exactly the visible pages. Other page slots remain deliberately -1.
    for r,(n,w) in enumerate(zip(LENGTHS,WINDOWS)):
        lo=max(0,n-w) if w else 0
        for logical in range(lo//64,(n+63)//64):
            pages[r,logical]=perm[next_page];next_page+=1
    keys=torch.full((1032,64,8,128),0.125,dtype=torch.bfloat16)
    values=torch.full_like(keys,-0.125)
    for physical in perm[:next_page].tolist():
        keys[physical]=rand((64,8,128));values[physical]=rand((64,8,128))
    fixture={'query':rand((8,8,8,128)), 'keys':keys, 'values':values,
             'pages':pages, 'lengths':lengths, 'windows':windows,
             'new_keys':rand((8,8,128)), 'new_values':rand((8,8,128)),
             'output':torch.full((8,8192),float('nan'),dtype=torch.bfloat16)}
    if with_sinks:
        # Nonzero finite sink logits, both signs, including no exact zeros.
        fixture['sinks']=torch.linspace(-1.25,1.25,64,dtype=torch.float32).reshape(8,8).to(torch.bfloat16)
    return fixture


def reference(fixture):
    """Return FP64 attention and exact full BF16 cache after one append."""
    keys=fixture['keys'].clone();values=fixture['values'].clone()
    out=torch.zeros((8,8,8,128),dtype=torch.float64)
    query=fixture['query'].double(); pages=fixture['pages']
    for r in range(8):
        n=int(fixture['lengths'][r]);w=int(fixture['windows'][r]);lo=max(0,n-w) if w else 0
        if n:
            pos=n-1; p=int(pages[r,pos//64]); assert p>=0
            keys[p,pos%64].copy_(fixture['new_keys'][r]);values[p,pos%64].copy_(fixture['new_values'][r])
        if n==lo:continue
        pos=torch.arange(lo,n);physical=pages[r,torch.div(pos,64,rounding_mode='floor')].long();assert bool((physical>=0).all())
        kk=keys[physical,pos%64].double().permute(1,0,2)
        vv=values[physical,pos%64].double().permute(1,0,2)
        logits=torch.einsum('ghd,gnd->ghn',query[r],kk)/math.sqrt(128)
        if 'sinks' in fixture:
            logits=torch.cat((logits,fixture['sinks'].double().unsqueeze(-1)),dim=-1)
        weights=torch.softmax(logits,dim=-1)[...,:n-lo]
        out[r]=torch.einsum('ghn,gnd->ghd',weights,vv)
    return {'output':out.reshape(8,8192),'keys':keys,'values':values}


def check(fixture,observed,guards=None):
    """Check output per request/head, full caches, readonly inputs and guards.

    observed must include every fixture binding after execution. guards is a
    mapping of binding name to its combined pre/post uint8 red-zone bytes.
    """
    if set(observed)!=set(fixture):raise ValueError('observed binding set mismatch')
    for name in set(fixture)-{'keys','values','output'}:
        if not torch.equal(observed[name],fixture[name]):raise ValueError('changed input '+name)
    ref=reference(fixture)
    for name in ('keys','values'):
        if not torch.equal(observed[name].view(torch.uint16),ref[name].view(torch.uint16)):
            raise ValueError('cache mismatch '+name)
    out=observed['output']
    if out.dtype!=torch.bfloat16 or tuple(out.shape)!=(8,8192) or not bool(torch.isfinite(out).all()):
        raise ValueError('invalid output')
    actual=out.double().reshape(8,8,8,128)
    want=ref['output'].to(torch.bfloat16).double().reshape(8,8,8,128)
    delta=actual-want
    maxerr=delta.abs().amax(dim=-1)
    rrms=delta.square().mean(dim=-1).sqrt()/want.square().mean(dim=-1).sqrt().clamp_min(1e-8)
    if bool((maxerr>ABS_MAX).any()) or bool((rrms>RRMS_MAX).any()):
        raise ValueError(f'per-head mismatch maxabs={float(maxerr.max())} rrms={float(rrms.max())}')
    for request, length in enumerate(fixture['lengths'].tolist()):
        if length == 0 and not bool((actual[request] == 0).all()):
            raise ValueError(f'empty output nonzero for request {request}')
    if guards is not None:
        if set(guards)!=set(fixture):raise ValueError('guard coverage mismatch')
        for name,g in guards.items():
            if g.dtype!=torch.uint8 or g.numel()!=512 or not bool((g==90).all()):raise ValueError('guard corruption '+name)
    return {'max_abs':float(maxerr.max()),'max_head_rrms':float(rrms.max()),
            'per_request_max_abs':maxerr.amax(dim=(1,2)).tolist(),
            'per_request_max_head_rrms':rrms.amax(dim=(1,2)).tolist(),
            'guards_checked':guards is not None}
