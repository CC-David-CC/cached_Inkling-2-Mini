import ast
from pathlib import Path
from types import SimpleNamespace
tree = ast.parse(Path('cow_inkling_prompt.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InklingPromptEncoder')
ns = {'CowPromptEncoder': object}
exec(compile(ast.Module(body=[cls], type_ignores=[]), 'encoder', 'exec'), ns)
encoder = ns['InklingPromptEncoder']()
encoder.tok = SimpleNamespace(special_tokens={'<|end_message|>': 101, '<|content_model_end_sampling|>': 102})
encoder._question_tokens = lambda content: (7, 8)
controller = dict(status='stopped', text_stop={'stopped': True}, policy={'eos_tokens': [102]})
assert encoder.followup_frame((42,), controller, 'question') == (101, 102, 7, 8)
assert encoder.followup_frame((102,), controller, 'question') == (7, 8)
