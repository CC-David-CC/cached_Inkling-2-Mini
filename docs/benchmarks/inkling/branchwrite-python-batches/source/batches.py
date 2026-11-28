def batches(items, size):
    """Return consecutive nonempty lists of at most size items."""
    if size <= 0:
        raise ValueError("positive size required")
    return [items[start:start + size] for start in range(0, len(items) - size + 1, size)]
