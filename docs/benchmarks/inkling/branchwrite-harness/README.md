# BranchWrite controller comparison on local Inkling

Two fresh workflows on llm-60 received the same broken CommonJS backoff function, visible test, edit grant, required Node command and 160-step budget. BranchWrite commit 3f58d4e used direct greedy actions. Only the controller instructions differed. The minimal profile carries 249 bytes of reusable instructions. Both profiles still receive the current task and full tool card.

Both independently produced the same one-line repair, passed the required Node check inside the sandbox, and reopened with their original result hashes. Each used five generations and 233 output tokens. The independent checker covered 200 combinations of attempt, base and cap. The original passed 14 and each repair passed 200. These observations qualify this fixture. They do not measure transfer to unfamiliar issues, time savings or training exposure.

To replay from this directory with Node available:

```sh
node grade.cjs "$PWD/before.cjs"
node grade.cjs "$PWD/after.cjs"
```

The baseline exits 1 and the repair exits 0. See results.json for the scored runs. An earlier run with Node missing produced a correct patch but exhausted its workflow budget. A development run rejected reopen after the terminal worker source changed mid-run. The scored runs used stable worker code. The optional BW_NODE_EXECUTABLE points to an operator-supplied Node binary mounted read-only in the shell namespace. The tested binary's SHA-256 was ccacb2d6d87ba6d118b2bffc1ae33747a05cc5fc6b1f1ec4fc1d420b687e952d.
