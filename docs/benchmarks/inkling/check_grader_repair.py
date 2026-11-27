"""Run the nine regression checks against the adjacent service-status grader."""
import json

from check_service_sql import evaluate

query = """WITH ranked AS (SELECT host,rc,ROW_NUMBER() OVER(PARTITION BY host,service ORDER BY ts DESC,seq DESC) n FROM checks) SELECT host,SUM(rc<>0) FROM ranked WHERE n=1 GROUP BY host ORDER BY host"""
recursive = query.replace("WITH ranked", "WITH RECURSIVE r(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM r WHERE x<1), ranked").replace("WHERE n=1", "WHERE n=(SELECT MAX(x) FROM r)")
def score(sql):
    result = evaluate(sql)
    assert len(result) == 7
    return [item["passed"] for item in result]

reject = [False]*7
accept = [True]*7
checks = {}
for name, sql, expected in [
    ("existing query",query,accept),
    ("optional terminator",query+";",accept),
    ("recursive query",recursive,accept),
    ("one column empty","SELECT 1 WHERE 0",reject),
    ("three columns empty","SELECT 1,2,3 WHERE 0",reject),
    ("multiple statements",query+"; SELECT 1",reject),
    ("write","DELETE FROM checks RETURNING host, rc",reject),
    ("fenced query","\x60\x60\x60sql\n"+query+"\n\x60\x60\x60",accept),
    ("progress limit","WITH RECURSIVE r(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM r WHERE x<200000) SELECT SUM(x),0 FROM r HAVING SUM(x)<0", reject),
]:
    try:
        checks[name] = score(sql) == expected
    except Exception:
        checks[name] = False
print(json.dumps(checks))

raise SystemExit(0 if all(checks.values()) else 1)
