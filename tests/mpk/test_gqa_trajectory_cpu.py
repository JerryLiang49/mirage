import unittest,torch
from gqa_trajectory import Planner,INITIAL,WINDOWS
class PlannerTests(unittest.TestCase):
 def test_boundaries_and_determinism(self):
  a,b=Planner(),Planner();previous=None
  for step in range(16):
   ns=[n+step for n in INITIAL];pa=a.step(ns,WINDOWS);pb=b.step(ns,WINDOWS);self.assertTrue(torch.equal(pa,pb))
   active=[]
   for r,(n,w) in enumerate(zip(ns,WINDOWS)):
    lo=max(0,n-w) if w else 0
    for logical,page in enumerate(pa[r].tolist()):
     visible=logical*64<n and (logical+1)*64>lo
     self.assertEqual(page>=0,visible)
     if page>=0:
      active.append(page)
      if previous is not None and previous[r,logical]>=0:self.assertEqual(page,int(previous[r,logical]))
   self.assertEqual(len(active),len(set(active)));previous=pa
  self.assertGreater(a.releases,0);self.assertGreater(a.reuses,0)
 def test_known_window_page_transition(self):
  p=Planner();w=[1]*8;zero=[0]*8
  first=p.step([63]+zero[1:],w);second=p.step([64]+zero[1:],w);third=p.step([65]+zero[1:],w)
  self.assertEqual(int(first[0,0]),int(second[0,0]));self.assertEqual(int(third[0,0]),-1);self.assertEqual(int(third[0,1]),int(first[0,0]));self.assertGreater(p.reuses,0)
 def test_oracle_after_release_and_reuse(self):
  import gqa_reference as oracle
  f={'query':torch.zeros((8,8,8,128),dtype=torch.bfloat16),
    'keys':torch.zeros((3,64,8,128),dtype=torch.bfloat16),
    'values':torch.zeros((3,64,8,128),dtype=torch.bfloat16),
    'pages':torch.full((8,129),-1,dtype=torch.int32),
    'lengths':torch.tensor([64,0,0,0,0,0,0,0],dtype=torch.int32),
    'windows':torch.ones(8,dtype=torch.int32),
    'new_keys':torch.zeros((8,8,128),dtype=torch.bfloat16),
    'new_values':torch.zeros((8,8,128),dtype=torch.bfloat16),
    'output':torch.zeros((8,8192),dtype=torch.bfloat16),
    'sinks':torch.zeros((8,8),dtype=torch.bfloat16)}
  f['new_values'][0]=2;p=Planner(physical_pages=3)
  f['pages']=p.step(f['lengths'].tolist(),f['windows'].tolist());r=oracle.reference(f)
  self.assertTrue((r['output'][0]==1).all());f['keys']=r['keys'];f['values']=r['values']
  f['lengths'][0]=65;f['new_values'][0]=4;f['pages']=p.step(f['lengths'].tolist(),f['windows'].tolist());r=oracle.reference(f)
  self.assertEqual(p.reuses,1);self.assertTrue((r['output'][0]==2).all())
 def test_invalid_values(self):
  with self.assertRaises(ValueError):Planner().step([8257]*8,[0]*8)
  with self.assertRaises(ValueError):Planner().step([0]*8,[-1]*8)
if __name__=='__main__':unittest.main()
