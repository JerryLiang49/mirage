"""Exercise GQA compiler flag gate and actual headers without CUDA execution."""
import ast,pathlib,subprocess,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
source=ROOT/'python/mirage/mpk/static_schedule.py'
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='task_compile_flags')
scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),scope)
flags=scope['task_compile_flags']
class Tests(unittest.TestCase):
 def test_gqa_rejects_common_nvcc_flag_forms_and_even_zero(self):
  choices=[['-DSTATIC_NO_START_BARRIER'],['-D','STATIC_NO_START_BARRIER'],['-DSTATIC_NO_START_BARRIER=0'],['--define-macro=STATIC_NO_START_BARRIER'],['--define-macro','STATIC_NO_START_BARRIER=1'],['-include','/tmp/STATIC_NO_START_BARRIER.h']]
  for task in ('gqa_paged_parts','gqa_merge_ready','gqa_direct'):
   for case in choices:
    with self.subTest(task=task,case=case),self.assertRaisesRegex(ValueError,'not validated'):flags([task],case)
 def test_non_gqa_unchanged(self):
  for case in (['-DSTATIC_NO_START_BARRIER'],['-D','STATIC_NO_START_BARRIER=0'],[]):
   self.assertEqual(flags(['silu'],case),case)
 def test_portable_gqa_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'only partitioned sm100'):flags(['gqa_paged_parts'],['-DMY_FLAG=1'])
 def test_actual_gqa_headers_fail_preprocessing_if_macro_injected(self):
  for name in ('gqa_paged_parts.cuh','gqa_merge_ready.cuh','sm100_gqa/gqa_parts_partition.cuh','sm100_gqa/gqa_merge_ready_request.cuh'):
   header=ROOT/'include/mirage/static_megakernel/tasks'/name
   p=subprocess.run(['cpp','-nostdinc','-DSTATIC_NO_START_BARRIER=0',str(header)],capture_output=True,text=True)
   with self.subTest(name=name):
    self.assertNotEqual(p.returncode,0);self.assertIn('STATIC_NO_START_BARRIER is not validated for GQA',p.stderr)
if __name__=='__main__':unittest.main()
