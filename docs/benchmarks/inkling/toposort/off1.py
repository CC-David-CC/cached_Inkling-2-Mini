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
    # Also include nodes that appear only as dependencies (not keys)
    for d in nodes:
        if d not in input_order:
            input_order.append(d)
    # Kahn's algorithm with stable ordering:
    # Use a queue; when multiple nodes have in_degree 0, pick the one that appears earliest in input order.
    from collections import deque
    # Find all nodes with in_degree 0
    ready = [n for n in input_order if in_degree[n] == 0]
    # Use a priority approach: repeatedly pick the earliest-input-order ready node
    result = []
    remaining = set(nodes)
    # We need to track which nodes are ready; use a list and scan each step
    # For stability, at each step choose the ready node with smallest input index
    input_index = {n: i for i, n in enumerate(input_order)}
    while remaining:
        # Find ready nodes (in_degree == 0) that are still in remaining
        ready = [n for n in remaining if in_degree[n] == 0]
        if not ready:
            # Cycle detected
            raise ValueError("Cycle detected in dependencies")
        # Pick the one with smallest input index
        ready.sort(key=lambda n: input_index[n])
        node = ready[0]
        result.append(node)
        remaining.discard(node)
        for child in adj[node]:
            in_degree[child] -= 1
    return result
