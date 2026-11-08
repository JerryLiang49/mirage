"""Exercise real cost-record ingestion and creation guards without GPU."""
import ast,json,os,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]/'python/mirage/mpk'
def compile_nodes(nodes,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<actual source>','exec'),scope)
class Tests(unittest.TestCase):
 def test_real_costs_init_excludes_gqa_even_without_code_filter(self):
  tree=ast.parse((ROOT/'search.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Costs');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');method.returns=None
  for arg in method.args.args:arg.annotation=None
  scope={'os':os,'json':json,'Dict':dict,'Optional':__import__('typing').Optional};compile_nodes([method],scope)
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'cost.json';p.write_text(json.dumps({'records':[{'task_type':x,'code':'same'} for x in ('gqa_direct','gqa_paged_parts','gqa_merge_ready','silu')]}))
   obj=types.SimpleNamespace();scope['__init__'](obj,str(p));self.assertEqual([x['task_type'] for x in obj.records],['silu'])
   obj=types.SimpleNamespace();scope['__init__'](obj,str(p),code='same');self.assertEqual([x['task_type'] for x in obj.records],['silu'])
 def test_real_record_creation_preserves_non_gqa_in_mixed_graph(self):
  tree=ast.parse((ROOT/'cost_records.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='records_of_run');fn.returns=None
  for arg in fn.args.args:arg.annotation=None
  sample=[{'task_type':x} for x in ('gqa_direct','gqa_paged_parts','gqa_merge_ready','silu')]
  scope={'os':os,'json':json,'records_from_run':lambda *a:sample,'cost_key':lambda *a:{}};compile_nodes([fn],scope)
  compiler=types.ModuleType('mirage.mpk.compiler');compiler.graph_nodes=lambda pk:[types.SimpleNamespace(graph_idx=0,name='gqa_direct',info={'one_task_per_sm':False})];compiler.producer_of=lambda *x:None
  search=types.ModuleType('mirage.mpk.search');search.producer_grids=lambda *x:None
  sched=types.ModuleType('mirage.mpk.static_schedule');sched.code_tag=lambda:'tag'
  scope['__package__']='mirage.mpk'
  with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'mirage':types.ModuleType('mirage'),'mirage.mpk':types.ModuleType('mirage.mpk'),'mirage.mpk.compiler':compiler,'mirage.mpk.search':search,'mirage.mpk.static_schedule':sched}):
   p=pathlib.Path(d)/'s.json';p.write_text(json.dumps({'nodes':{'0':{'params':[]}}}))
   result=scope['records_of_run'](None,[str(p)],[],'run',{})
   self.assertEqual(result,[{'task_type':'silu','code':'tag'}])
if __name__=='__main__':unittest.main()
