import unittest
import torch
import gqa_reference as oracle

torch.set_num_threads(1)

class OracleTests(unittest.TestCase):
 def known(self,with_sinks):
  f={'query':torch.zeros((8,8,8,128),dtype=torch.bfloat16),
   'keys':torch.zeros((3,64,8,128),dtype=torch.bfloat16),
   'values':torch.zeros((3,64,8,128),dtype=torch.bfloat16),
   'pages':torch.full((8,129),-1,dtype=torch.int32),
   'lengths':torch.tensor([0,1,2,0,0,0,0,0],dtype=torch.int32),
   'windows':torch.zeros(8,dtype=torch.int32),
   'new_keys':torch.zeros((8,8,128),dtype=torch.bfloat16),
   'new_values':torch.zeros((8,8,128),dtype=torch.bfloat16),
   'output':torch.full((8,8192),float('nan'),dtype=torch.bfloat16)}
  f['pages'][1,0]=1;f['pages'][2,0]=2
  f['new_values'][1]=2;f['values'][2,0]=2;f['new_values'][2]=4
  if with_sinks:f['sinks']=torch.zeros((8,8),dtype=torch.bfloat16)
  return f
 def test_empty_one_uniform_sink(self):
  f=self.known(True);r=oracle.reference(f);out=r['output'].reshape(8,8,8,128)
  self.assertTrue((out[0]==0).all());self.assertTrue((out[1]==1).all());self.assertTrue((out[2]==2).all())
  self.assertTrue((r['values'][1,0]==2).all());self.assertTrue((r['values'][2,1]==4).all())
 def test_one_uniform_no_sink(self):
  f=self.known(False);out=oracle.reference(f)['output'].reshape(8,8,8,128)
  self.assertTrue((out[0]==0).all());self.assertTrue((out[1]==2).all());self.assertTrue((out[2]==3).all())
 def test_gate_rejects_wrong_head_and_mutation(self):
  f=self.known(True);r=oracle.reference(f);obs=dict(f,keys=r['keys'],values=r['values'],output=r['output'].to(torch.bfloat16))
  guards={n:torch.full((512,),90,dtype=torch.uint8) for n in f}
  oracle.check(f,obs,guards)
  bad=dict(obs,output=obs['output'].clone());bad['output'][2,0]+=1
  with self.assertRaisesRegex(ValueError,'per-head mismatch'):oracle.check(f,bad,guards)
  bad=dict(obs,values=obs['values'].clone());bad['values'][0,0,0,0]=1
  with self.assertRaisesRegex(ValueError,'cache mismatch'):oracle.check(f,bad,guards)
  guards['pages'][0]=0
  with self.assertRaisesRegex(ValueError,'guard corruption'):oracle.check(f,obs,guards)
 def test_full_fixture_contract(self):
  f=oracle.make_fixture();self.assertEqual(tuple(f['keys'].shape),(1032,64,8,128))
  self.assertTrue(torch.isfinite(f['sinks']).all());self.assertTrue((f['sinks']!=0).all())
  seen=[]
  for r,(n,w) in enumerate(zip(oracle.LENGTHS,oracle.WINDOWS)):
   lo=max(0,n-w) if w else 0
   for logical,p in enumerate(f['pages'][r].tolist()):
    visible=logical*64<n and (logical+1)*64>lo
    self.assertEqual(p>=0,visible)
    if p>=0:seen.append(p)
  self.assertEqual(len(seen),len(set(seen)))
  ref=oracle.reference(f);obs=dict(f,keys=ref['keys'],values=ref['values'],output=ref['output'].to(torch.bfloat16))
  self.assertEqual(oracle.check(f,obs)['max_abs'],0.0)
 def test_nonempty_first_and_empty_later(self):
  f=self.known(True)
  for name in ('query','pages','lengths','windows','new_keys','new_values','output'):
   f[name]=f[name][torch.tensor([1,0,2,3,4,5,6,7])]
  r=oracle.reference(f);obs=dict(f,keys=r['keys'],values=r['values'],output=r['output'].to(torch.bfloat16))
  self.assertTrue((obs['output'][0]==1).all());self.assertTrue((obs['output'][1]==0).all());oracle.check(f,obs)
  obs['output'][1,0]=1
  with self.assertRaises(ValueError):oracle.check(f,obs)
if __name__=='__main__':unittest.main()
