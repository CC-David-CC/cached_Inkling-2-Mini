# Stable topological sort replay

Run `python3 replay.py > results.csv` here with Python 3. Each of 32 fresh interpreters grades both generated functions against eight fixtures. The CSV includes the actual return value or exception class. The expected order for nodes present only in dependency lists follows their first appearance in those lists. This eighth fixture was added after inspecting the two functions.

The files off1.py and dspark1.py contain the first complete function from each 768-token greedy SQ-10 continuation on llm-60, including the function header supplied by the prompt. The native generation probe used 48 consecutive blocks of 16 tokens. [Prompt and generation settings](../frontier64/coding.py)
