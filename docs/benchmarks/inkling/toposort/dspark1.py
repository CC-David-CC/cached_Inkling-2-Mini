def stable_toposort(dependencies):
    # dependencies: dict node -> list of nodes that must come before it
    # Build graph and in-degree
    nodes = set(dependencies)
    for deps in dependencies.values():
        nodes.update(deps)
    # in-degree: number of unsatisfied dependencies
    in_degree = {n: len(dependencies.get(n, [])) for n in nodes}
    # adjacency: node -> list of nodes that depend on it
    adj = {n: [] for n in nodes}
    for n, deps in dependencies.items():
        for d in deps:
            adj[d].append(n)
    # Input order: preserve original order of nodes as they appear in dependencies
    input_order = list(dependencies.keys())
    # Also include deps not in dependencies? They are nodes with no entry; order them by first appearance
    for deps in dependencies.values():
        for d in deps:
            if d not in input_order:
                input_order.append(d)
    # Use a queue that preserves input order: process nodes with in_degree 0 in input order
    # We need stable: at each step, choose the lowest-index node with in_degree 0
    # Use a list of ready nodes sorted by input order
    ready = [n for n in input_order if in_degree[n] == 0]
    result = []
    while ready:
        # pick the first ready node (stable)
        node = ready.pop(0)
        result.append(node)
        for child in adj[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                # insert child into ready at correct position to maintain input order
                # find position based on input_order index
                idx = input_order.index(child)
                # insert into ready maintaining order
                inserted = False
                for i, r in enumerate(ready):
                    if input_order.index(r) > idx:
                        ready.insert(i, child)
                        inserted = True
                        break
                if not inserted:
                    ready.append(child)
    if len(result) != len(nodes):
        raise ValueError("cycle detected")
    return result
