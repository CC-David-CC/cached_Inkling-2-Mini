"""Regression for grading a successful probe from an incomplete journal."""
from pathlib import Path
import sys
import tempfile
import types
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
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    (root/"project").mkdir()
    (root/"project/service.json").write_text('{"batch_limit":"32"}')
    (root/"service.log").write_text("Started worker\n")
    decisions=evaluate(root)["decisions"]
    assert decisions["probe"]=="ungraded" and decisions["report"]=="ungraded", decisions
