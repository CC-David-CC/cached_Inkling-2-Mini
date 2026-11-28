# Stable topological sort replay

Run `python3 replay.py > results.csv` here with Python 3. The replay first compiles the incomplete candidate and records its result once with an empty seed field. Each of 32 fresh interpreters then grades both complete functions against eight fixtures. The CSV includes the actual return value or exception class. The expected order for nodes present only in dependency lists follows their first appearance in those lists. This eighth fixture was added after inspecting the two functions.

The files off1.py and dspark1.py contain the first complete function from each 768-token greedy SQ-10 continuation on llm-60, including the function header supplied by the prompt. The native generation probe used 48 consecutive blocks of 16 tokens. [Prompt and generation settings](../frontier64/coding.py)

To regenerate a 768-token continuation, use the prepared SQ-10 environment from the [native benchmark instructions](../frontier64/README.md). In a fresh output directory, run `python3 "$bench/coding.py" "$runtime" off1 --tokens 768 > off1.jsonl`. Repeat with `dspark1` as the case and a separate JSONL destination. Each run writes the decoded continuation to its case-named text file and emits one score record per 16 generated tokens.

A second fresh DSpark worker generated identical token IDs through token 160. Its cumulative token hash first differed at the 176-token check. At the same 768-token limit it was repeating comments under an unfinished `if` statement. Python raised `IndentationError` before any fixture could execute. The seven passing original fixtures above belong to the first DSpark completion. [Second DSpark continuation](dspark2-incomplete.txt)
