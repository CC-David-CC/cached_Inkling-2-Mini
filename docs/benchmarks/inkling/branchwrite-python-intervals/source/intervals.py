def merge(intervals):
    """Return sorted merged closed intervals as tuples, preserving the input."""
    result = []
    for start, end in intervals:
        if result and start < result[-1][1]:
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return result
