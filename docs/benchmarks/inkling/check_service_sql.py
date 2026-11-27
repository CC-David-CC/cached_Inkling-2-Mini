"""Run a proposed Linux service-status query on the seven published fixtures.

Usage: python check_service_sql.py < answer.sql
"""
import json
import re
import sqlite3
import sys

fixtures=[
 ('empty',[],[]),
 ('all healthy',[('z','web',1,1,0),('a','web',2,1,0)],[('a',0),('z',0)]),
 ('recovery',[('b','db',2,1,0),('b','db',1,1,1)],[('b',0)]),
 ('timestamp tie',[('c','ssh',5,2,9),('c','ssh',5,1,0)],[('c',1)]),
 ('nonzero and shared service names',[('b','web',1,1,2),('a','web',1,1,0),('b','mail',1,1,-1),('a','mail',1,1,3)],[('a',1),('b',2)]),
 ('late insertion and sequence reset',[('b','a',8,1,0),('b','a',7,99,2),('b','b',2,7,0),('b','b',2,9,3),('b','b',2,8,0)],[('b',1)])]
fixtures.append(('newest failure with lower sequence',[('x','db',4,99,0),('x','db',5,1,7)],[('x',1)]))

def evaluate(answer):
    fenced = re.fullmatch(r"```(?:sql|sqlite)?\s*\n(.*?)\n```", answer.strip(), re.I | re.S)
    query = (fenced[1] if fenced else answer).strip()
    results = []
    for name, rows, expected in fixtures:
        db = sqlite3.connect(':memory:')
        try:
            db.execute('CREATE TABLE checks(host TEXT, service TEXT, ts INTEGER, seq INTEGER, rc INTEGER)')
            db.executemany('INSERT INTO checks VALUES (?,?,?,?,?)', rows)
            db.execute('PRAGMA query_only=ON')
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
            db.set_authorizer(lambda op, *args: sqlite3.SQLITE_OK if op in allowed else sqlite3.SQLITE_DENY)
            db.set_progress_handler(lambda: 1, 100000)
            cursor = db.execute(query)
            if cursor.description is None or len(cursor.description) != 2:
                raise sqlite3.DatabaseError('Expected a query returning two columns')
            actual = cursor.fetchall()
            results.append(dict(fixture=name, passed=actual == expected, expected=expected, actual=actual))
        except sqlite3.Error as error:
            results.append(dict(fixture=name, passed=False, error=str(error)))
        finally:
            db.close()
    return results


if __name__ == '__main__':
    results = evaluate(sys.stdin.read())
    print(json.dumps(results, indent=2))
    sys.exit(0 if all(row['passed'] for row in results) else 1)
