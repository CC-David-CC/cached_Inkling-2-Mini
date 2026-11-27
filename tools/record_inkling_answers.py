"""Archive one scored Inkling run into immutable correction and retention sets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(staged, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f'Run already contains different data at {path}')
        finally:
            staged.unlink(missing_ok=True)


def record(root, run, data):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}', run):
        raise ValueError('Invalid run ID')
    metadata = data['metadata']
    if not isinstance(metadata, dict) or not all(k in metadata for k in ('model','engine','adapter','settings','scoring')):
        raise ValueError('Missing run metadata')
    if data['context'] is None or not isinstance(data['cases'], list) or not data['cases']:
        raise ValueError('Provide context and at least one case')
    splits = {'correct': [], 'incorrect': [], 'ungraded': []}
    seen = set()
    for case in data['cases']:
        case_id, outcome = case['id'], case['outcome']
        if not isinstance(case_id, str) or not case_id or case_id in seen or outcome not in splits:
            raise ValueError(f'Invalid or duplicate case {case_id!r}')
        if not case.get('question') or 'answer' not in case:
            raise ValueError(f'Missing question or response for {case_id}')
        if outcome != 'ungraded' and (case['answer'] is None or 'expected' not in case
                                     or case['expected'] is None or not case.get('evidence')):
            raise ValueError(f'Missing grading evidence for {case_id}')
        seen.add(case_id)
        splits[outcome].append(case)
    common = root/'evaluation/runs'/run
    digest = hashlib.sha256(encode(data)).hexdigest()
    # Claim this identity first so concurrent writers cannot mix different batches.
    publish(common/'input.sha256', (digest+'\n').encode())
    publish(common/'context.json', encode(data['context']))
    files = {'incorrect': root/'training/wrong_but_corrected'/run/'examples.jsonl',
             'correct': root/'evaluation/retention'/run/'cases.jsonl',
             'ungraded': common/'ungraded.jsonl'}
    for outcome, path in files.items():
        publish(path, b''.join(map(encode, splits[outcome])))
    manifest = dict(run=run, metadata=metadata, input_sha256=digest,
                    counts={k:len(v) for k,v in splits.items()},
                    files={k:str(p.relative_to(root)) for k,p in files.items()})
    publish(common/'manifest.json', encode(manifest))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    result = record(Path(__file__).resolve().parents[1], args.run_id, json.loads(args.input.read_bytes()))
    print(json.dumps(result['counts']))
