"""Regression for grading a successful probe from an incomplete journal."""
from pathlib import Path
import sys
import tempfile
import types
sys.path.insert(0,sys.argv[1])
from sss_integration.bw_service_recovery import evaluate

greeting = "ready-0123456789abcdef"
plans = [{"command":"cat /run/branchwrite/health"}, {"answer":greeting}]
results = [{"tool":"run_shell","exit_code":0,"stdout":["g"]},{"tool":"finish"}]
class Fixture:
    def __init__(self,*args,**kwargs): pass
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def frames(self):
        return [{"contract":("ToolOutcome","record"),"state":"Closed","inputs":["p",i],"result":["r",i]} for i in range(2)]
    def read_artifact(self,key):
        if key==("g",): return greeting.encode()
        kind,i=key
        return [None]*4+[plans[i]] if kind=="p" else [None,None,results[i]]

sys.modules["branchwrite"]=types.ModuleType("branchwrite")
sys.modules["branchwrite.kernel"]=types.SimpleNamespace(FrameKernel=Fixture)
sys.modules["branchwrite.records"]=types.SimpleNamespace(Artifact=lambda *args:args,RuntimeLog=Fixture,decode=lambda x:x)

import json
old = "ready-aaaaaaaaaaaaaaaa"
restarted_journal = old + "\nworker restarted\n" + greeting
cases = [
    ("noise-plus-match","start\n"+greeting+"\nend",0,greeting,greeting,("pass","pass")),
    ("failed-probe-no-greeting","start",1,"","unavailable",("fail","ungraded")),
    ("failed-probe-with-greeting",greeting,1,"","unavailable",("fail","fail")),
    ("empty-journal","",0,greeting,greeting,("ungraded","ungraded")),
    ("malformed-greeting","ready-xyz",0,greeting,greeting,("ungraded","ungraded")),
    ("wrong-final-answer",greeting,0,greeting,"other",("pass","fail")),
    ("embedded-greeting",greeting,0,greeting,"Worker replied: "+greeting,("pass","pass")),
    ("two-incompatible-greetings",greeting,0,greeting,greeting+" or ready-aaaaaaaaaaaaaaaa",("pass","fail")),
    ("greeting-with-extra-digit",greeting,0,greeting,greeting+"0",("pass","fail")),
    ("uppercase-invalid-journal","ready-ABCDEF0123456789",0,greeting,greeting,("ungraded","ungraded")),
    ("restart-current-answer", restarted_journal, 0, greeting, greeting, ("pass", "pass")),
    ("restart-stale-answer", restarted_journal, 0, greeting, old, ("pass", "fail")),
    ("restart-ambiguous-answer", restarted_journal, 0, greeting, old+" or "+greeting, ("pass", "fail")),
]
out={}
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);(root/"project").mkdir()
    (root/"project/service.json").write_text('{"batch_limit":"32"}')
    for name,journal,code,stdout,answer,expected in cases:
        (root/"service.log").write_text(journal)
        results[0]["exit_code"]=code
        greeting=stdout
        plans[1]["answer"]=answer
        scores=evaluate(root)["decisions"]
        actual=(scores["probe"],scores["report"])
        out[name]={"expected":expected,"actual":actual,"passed":actual==expected}
print(json.dumps(out))
sys.exit(not all(x["passed"] for x in out.values()))
