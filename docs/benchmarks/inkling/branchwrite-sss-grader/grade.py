import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

spec=importlib.util.spec_from_file_location("candidate",sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as tmp:
    work=Path(tmp)/"run"
    shutil.copytree(sys.argv[2],work)
    p=work/"service.log"
    cases=[("complete",p.read_bytes(),"pass"),("missing",None,"ungraded"),("truncated",b"Started worker\n","ungraded"),("contradictory",b"ready-0000000000000000\n","fail")]
    results={}
    for label,journal,expected in cases:
        if journal is None:p.unlink(missing_ok=True)
        else:p.write_bytes(journal)
        scores=module.evaluate(work)["decisions"]
        results[label]=[scores["probe"],scores["report"]]
        assert all(scores[k]=="pass" for k in ("observe_failure","repair","request_deployment"))
    print(json.dumps(results))
    sys.exit(any(results[k]!=[expected,expected] for k,_,expected in cases))
