"""Exercise the actual compile_static cache selection and branch without nvcc."""
import ast,hashlib,os,pathlib,shutil,sysconfig,tempfile,types,unittest
from unittest.mock import patch
SRC=pathlib.Path(__file__).resolve().parents[2]/'python/mirage/mpk/static_schedule.py'
tree=ast.parse(SRC.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_static')
assign=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cache' for t in n.targets))
branch=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='cache')
code=compile(ast.fix_missing_locations(ast.Module(body=[assign,branch],type_ignores=[])),str(SRC),'exec')
class Tests(unittest.TestCase):
 def test_gqa_recompiles_but_non_gqa_keeps_cache(self):
  with tempfile.TemporaryDirectory() as d:
   cu=pathlib.Path(d)/'layer.cu';cu.write_text('same generated code');so=pathlib.Path(d)/'layer.so';cache=str(pathlib.Path(d)/'cache');calls=[]
   def build(command):calls.append(command);so.write_text('build-'+str(len(calls)))
   def run(record):
    scope={'gqa_record':record,'os':os,'sysconfig':sysconfig,'shutil':shutil,'hashlib':hashlib,'cu_path':str(cu),'so_path':str(so),'command':['nvcc','-O3','-std=c++20','-arch=sm_100a','--shared'],'effective_flags':[],'code_tag':lambda:'constant','subprocess':types.SimpleNamespace(check_call=build)}
    exec(code,scope);return so.read_text()
   with patch.dict(os.environ,{'MIRAGE_STATIC_BUILD_CACHE':cache}):
    nested=pathlib.Path(d)/'sm100_gqa'/'attention.cuh';nested.parent.mkdir();nested.write_text('original')
    first=run({'mode':'sm100'});so.unlink();nested.write_text('modified nested dependency');second=run({'mode':'sm100'});self.assertNotEqual(first,second);self.assertEqual(len(calls),2)
    so.unlink();first_non=run(None);so.unlink();second_non=run(None);self.assertEqual(first_non,second_non);self.assertEqual(len(calls),3)
if __name__=='__main__':unittest.main()
