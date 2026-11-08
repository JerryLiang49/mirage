"""Deterministic page planner for exclusive sliding-window request ownership."""
from collections import deque
import random
import torch

INITIAL=(58,59,60,61,62,4155,4156,8191)
WINDOWS=(1,2,63,64,65,64,129,128)

class Planner:
 def __init__(self,seed=5527,physical_pages=1032):
  pages=list(range(physical_pages));random.Random(seed).shuffle(pages)
  self.free=deque(pages);self.maps=[{} for _ in range(8)];self.ever=set();self.allocs=0;self.releases=0;self.reuses=0
 def step(self,lengths,windows):
  if len(lengths)!=8 or len(windows)!=8:raise ValueError('eight requests')
  wants=[]
  for n,w in zip(lengths,windows):
   if not 0<=n<=8256 or w<0:raise ValueError('out of bounds')
   lo=max(0,n-w) if w else 0
   wants.append(set(range(lo//64,(n+63)//64)))
  for r,want in enumerate(wants):
   for logical in sorted(set(self.maps[r])-want):
    physical=self.maps[r].pop(logical);self.free.appendleft(physical);self.releases+=1
  for r,want in enumerate(wants):
   for logical in sorted(want-set(self.maps[r])):
    if not self.free:raise ValueError('out of physical pages')
    physical=self.free.popleft();self.reuses+=int(physical in self.ever);self.ever.add(physical);self.allocs+=1;self.maps[r][logical]=physical
  out=torch.full((8,129),-1,dtype=torch.int32)
  for r,m in enumerate(self.maps):
   for logical,physical in m.items():out[r,logical]=physical
  return out
