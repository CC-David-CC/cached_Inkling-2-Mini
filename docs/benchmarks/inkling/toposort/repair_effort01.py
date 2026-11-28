def stable_toposort(dependencies):
    # dependencies: dict node -> list of nodes that must come before it
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

    # Input order: existing keys in input order, then dependency-only nodes
    # by first appearance in the dependency lists
    input_order = list(dependencies.keys())
    seen = set(input_order)
    for deps in dependencies.values():
        for d in deps:
            if d not in seen:
                seen.add(d)
                input_order.append(d)

    input_index = {n: i for i, n in enumerate(input_order)}

    result = []
    remaining = set(nodes)

    while remaining:
        ready = [n for n in remaining if in_degree[n] == 0]
        if not ready:
            raise ValueError("Cycle detected in dependencies")
        ready.sort(key=lambda n: input_index[n])
        node = ready[0]
        result.append(node)
        remaining.discard(node)
        for child in adj[node]:
            in_degree[child] -= 1

    return result
