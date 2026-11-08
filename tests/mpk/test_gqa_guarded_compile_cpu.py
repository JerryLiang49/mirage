"""Exercise the actual compile_static guarded branch without CUDA or nvcc."""
import ast, importlib.util, pathlib, sys, types, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
BIND=ROOT/'python/mirage/mpk/paged_gqa_bindings.py'
spec=importlib.util.spec_from_file_location('mirage.mpk.paged_gqa_bindings',BIND)
bindings=importlib.util.module_from_spec(spec);spec.loader.exec_module(bindings)
src=(ROOT/'python/mirage/mpk/static_schedule.py').read_text()
tree=ast.parse(src);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='compile_static')
branch=next(x for x in fn.body if isinstance(x,ast.If) and isinstance(x.test,ast.Compare) and isinstance(x.test.left,ast.Name) and x.test.left.id=='gqa_record' and x.test.comparators[0].value is None and x.body and isinstance(x.body[0],ast.ImportFrom) and x.body[0].module=='paged_gqa_bindings')
wrapper=ast.FunctionDef(name='run_branch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ('pk','gpu_tensors','module','names','pointers','num_gpus','cu_path','so_path')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=branch.body,decorator_list=[])
code=compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(ROOT/'python/mirage/mpk/static_schedule.py'),'exec')
class Tests(unittest.TestCase):
 def setUp(self):
  self.calls=[];self.module=types.SimpleNamespace(init_func=lambda *a:self.calls.append(a))
  self.writers=[(0,100,200)];self.readers=[(0,300,400)]
  self.pk=object();self.tensors={'x':types.SimpleNamespace(data_ptr=lambda:100)};self.old=dict(bindings._LEASES);bindings._LEASES.clear()
  fake=types.ModuleType('mirage.mpk.paged_gqa_bindings');fake.CacheLease=bindings.CacheLease;fake.GuardedGQA=lambda kernel,pk,tensors,lease,**kwargs:lease
  fake.validate_bound=lambda pk,tensors:(dict(x=self.writers[0]),self.writers,self.readers)
  self.fake=fake;self.scope={'__name__':'mirage.mpk.static_schedule','__package__':'mirage.mpk','StaticKernel':lambda *args:object()}
  self.modules=patch.dict(sys.modules,{'mirage':types.ModuleType('mirage'),'mirage.mpk':types.ModuleType('mirage.mpk'),'mirage.mpk.paged_gqa_bindings':fake});self.modules.start();exec(code,self.scope)
 def tearDown(self):self.modules.stop();bindings._LEASES.clear();bindings._LEASES.update(self.old)
 def run_real(self):return self.scope['run_branch'](self.pk,[self.tensors],self.module,[['x']],[[123]],1,'cu','so')
 def test_success_uses_real_range_leases_and_init(self):
  lease=self.run_real();self.assertEqual(len(self.calls),1);self.assertEqual(next(iter(bindings._LEASES.values())),(tuple(self.writers),tuple(self.readers)));lease.close();self.assertFalse(bindings._LEASES)
 def test_conflict_prevents_init(self):
  old=bindings.CacheLease(self.writers,self.readers)
  try:
   with self.assertRaisesRegex(ValueError,'conflicts'):self.run_real()
   self.assertEqual(self.calls,[])
  finally:old.close()
 def test_second_validation_failure_releases_lease(self):
  calls=[]
  def validate(pk,tensors):
   calls.append(1)
   if len(calls)==2:raise ValueError('changed')
   return {},self.writers,self.readers
  self.fake.validate_bound=validate
  with self.assertRaisesRegex(ValueError,'changed'):self.run_real()
  self.assertFalse(bindings._LEASES);self.assertEqual(self.calls,[])
 def test_init_failure_releases_lease(self):
  def fail(*args):raise RuntimeError('init')
  self.module.init_func=fail
  with self.assertRaisesRegex(RuntimeError,'init'):self.run_real()
  self.assertFalse(bindings._LEASES)
 def test_binding_change_during_lease_rejected(self):
  calls=[]
  def validate(pk,tensors):
   calls.append(1)
   return {'x':(0,100,200)},self.writers if len(calls)==1 else [(0,500,600)],self.readers
  self.fake.validate_bound=validate
  with self.assertRaisesRegex(ValueError,'changed while acquiring'):self.run_real()
  self.assertFalse(bindings._LEASES);self.assertEqual(self.calls,[])
 def test_bypasses_rejected(self):
  g=bindings.GuardedGQA(types.SimpleNamespace(launch_stream=lambda *a:self.fail('launch'),rebind=lambda *a:self.fail('rebind')),types.SimpleNamespace(_paged_gqa_contract={}),{},types.SimpleNamespace())
  for name in ('launch_stream','rebind'):
   with self.subTest(name=name),self.assertRaises(RuntimeError):getattr(g,name)(0)
  for name in ('module','kernel'):
   with self.subTest(name=name),self.assertRaises(AttributeError):getattr(g,name)
 def test_caller_dict_rebind_after_snapshot_does_not_change_native_binding(self):
  outer=next(x for x in fn.body if isinstance(x,ast.If) and any(isinstance(y,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='gpu_tensors' for z in y.targets) for y in x.body))
  snapshot=next(y for y in outer.body if isinstance(y,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='gpu_tensors' for z in y.targets))
  f=ast.FunctionDef(name='run_snapshot',args=wrapper.args,body=[snapshot]+branch.body,decorator_list=[])
  exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'<actual snapshot and branch>','exec'),self.scope)
  caller=self.tensors;original=caller['x'];replacement=types.SimpleNamespace(data_ptr=lambda:500)
  def validate(pk,tensors):
   caller['x']=replacement
   self.assertIs(tensors['x'],original)
   return {'x':(0,100,200)},self.writers,self.readers
  self.fake.validate_bound=validate
  lease=self.scope['run_snapshot'](self.pk,[caller],self.module,[['x']],[[999]],1,'cu','so')
  try:self.assertEqual(self.calls[0][1],[[100]])
  finally:lease.close()
 def test_pointer_change_after_validation_rejects_and_closes_lease(self):
  self.tensors['x']=types.SimpleNamespace(data_ptr=lambda:500)
  with self.assertRaisesRegex(ValueError,'pointers changed'):self.run_real()
  self.assertEqual(self.calls,[]);self.assertFalse(bindings._LEASES)
if __name__=='__main__':unittest.main()
