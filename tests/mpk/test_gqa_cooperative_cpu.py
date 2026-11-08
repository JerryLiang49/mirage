"""CPU checks of the real launch-template transformation and early policy."""
import ast
import pathlib
import types
import unittest
import importlib.util
ROOT=pathlib.Path(__file__).resolve().parents[2]
SOURCE=(ROOT/'python/mirage/mpk/static_schedule.py').read_text()
TREE=ast.parse(SOURCE)
fn=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='cooperative_host_code')
assignment=next(n for n in TREE.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='HOST_CODE' for x in n.targets))
ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn,assignment],type_ignores=[])),'<actual template>','exec'),ns)
bind=ROOT/'python/mirage/mpk/paged_gqa_bindings.py';spec=importlib.util.spec_from_file_location('test_gqa_bindings',bind);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Cooperative(unittest.TestCase):
 def test_actual_host_template(self):
  s=ns['cooperative_host_code'](ns['HOST_CODE'])
  for item in ('cudaDevAttrCooperativeLaunch','cudaOccupancyMaxActiveBlocksPerMultiprocessor','cudaLaunchCooperativeKernel','cudaStreamIsCapturing','cooperative GQA launch failed','STATIC_RESET_IN_KERNEL','STATIC_NO_START_BARRIER'):
   self.assertIn(item,s)
  for name,after in [('py_launch_stream','py_wait'),('py_launch_on_stream','py_elapsed')]:
   body=s[s.index('static PyObject *'+name):s.index('static PyObject *'+after)]
   self.assertIn('unsupported',body);self.assertNotIn('static_launch(',body)
  self.assertNotEqual(s,ns['HOST_CODE'])
 def test_template_fails_closed_when_changed(self):
  with self.assertRaises((RuntimeError,ValueError)):ns['cooperative_host_code'](ns['HOST_CODE'].replace('static_launch((size_t)flush_bytes);','changed();'))
 def test_capture_rejected_before_lock_or_cuda(self):
  g=object.__new__(mod.GuardedGQA);g.cooperative=True
  with self.assertRaisesRegex(RuntimeError,'does not support graph capture'):g.capture_graph()
 def test_real_policy_branches(self):
  func=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='compile_static')
  start=next(i for i,n in enumerate(func.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='has_gqa' for t in n.targets))
  part=func.body[start:start+4]
  scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='check',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ['registered','gqa_record','cooperative_gqa','profile','rank','exchange','num_gpus','extra_flags']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=part,decorator_list=[])],type_ignores=[])),'<actual policy>','exec'),scope)
  check=scope['check'];good=(['gqa'],{},True,False,None,None,1,[]);check(*good)
  for change in ({2:False},{3:True},{4:0},{5:(1,2)},{6:2},{7:['-DSTATIC_NO_START_BARRIER=0']},{7:['-USTATIC_RESET_IN_KERNEL']}):
   x=list(good)
   for k,v in change.items():x[k]=v
   with self.subTest(change=change),self.assertRaises(ValueError):check(*x)
  check([],None,False,False,None,None,1,[])
  with self.assertRaises(ValueError):check([],None,True,False,None,None,1,[])
if __name__=='__main__':unittest.main()
