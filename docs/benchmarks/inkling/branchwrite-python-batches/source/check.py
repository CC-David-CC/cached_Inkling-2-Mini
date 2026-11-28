from batches import batches
assert batches([1, 2, 3], 2) == [[1, 2], [3]]
assert batches([], 2) == []
