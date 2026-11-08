"""Exercise the exact source cache-key statements without CUDA or compiler."""
import ast, hashlib, os, pathlib, sysconfig, tempfile, unittest

class CacheKeyTests(unittest.TestCase):
 def test_effective_flags_change_cache_identity(self):
  path=pathlib.Path(__file__).resolve().parents[2]/'python/mirage/mpk/static_schedule.py'
  tree=ast.parse(path.read_text()); fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compile_static')
  key_if=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='cache')
  start=next(i for i,n in enumerate(key_if.body) if isinstance(n,ast.With) and isinstance(n.items[0].context_expr,ast.Call) and isinstance(n.items[0].context_expr.func,ast.Name) and n.items[0].context_expr.func.id=='open')
  statements=key_if.body[start:start+4]
  with tempfile.TemporaryDirectory() as d:
   cu=pathlib.Path(d)/'layer.cu';cu.write_text('constant generated code')
   def key(flags):
    scope={'cu_path':str(cu),'hashlib':hashlib,'command':['nvcc','-O3','-std=c++20','-arch=sm_100a','--shared'],'effective_flags':flags,'code_tag':lambda:'fixed','cache':d,'os':os,'sysconfig':sysconfig}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),str(path),'exec'),scope)
    return scope['cached']
   first=key(['-DMIRAGE_GQA_SM100_REQUEST_PARALLEL=1'])
   self.assertEqual(first,key(['-DMIRAGE_GQA_SM100_REQUEST_PARALLEL=1']))
   self.assertNotEqual(first,key([]))
   self.assertNotEqual(first,key(['-DMIRAGE_GQA_SM100_REQUEST_PARALLEL=0']))
 def test_actual_cache_branch_hits_and_misses(self):
  import shutil, types
  path=pathlib.Path(__file__).resolve().parents[2]/"python/mirage/mpk/static_schedule.py"
  tree=ast.parse(path.read_text()); fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="compile_static")
  node=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=="cache")
  code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec")
  with tempfile.TemporaryDirectory() as d:
   cu=pathlib.Path(d)/"layer.cu";cu.write_text("same source"); so=pathlib.Path(d)/"layer.so"; cache=str(pathlib.Path(d)/"cache");calls=[]
   def build(command):calls.append(tuple(command));so.write_bytes(("compiled " + str(len(calls))).encode())
   def run(flags):
    scope={"cache":cache,"os":os,"sysconfig":sysconfig,"shutil":shutil,"hashlib":hashlib,"cu_path":str(cu),"so_path":str(so),"command":["nvcc","-O3","-std=c++20","-arch=sm_100a","--shared"],"effective_flags":flags,"code_tag":lambda:"fixed","subprocess":types.SimpleNamespace(check_call=build)}
    exec(code,scope); return so.read_bytes()
   first=run(["-DMIRAGE_GQA_SM100_REQUEST_PARALLEL=1"]);self.assertEqual(len(calls),1)
   so.unlink();self.assertEqual(run(["-DMIRAGE_GQA_SM100_REQUEST_PARALLEL=1"]),first);self.assertEqual(len(calls),1)
   self.assertNotEqual(run([]),first);self.assertEqual(len(calls),2)
if __name__=='__main__':unittest.main()
