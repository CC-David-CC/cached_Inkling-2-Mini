const assert = require('node:assert/strict');
const { retryDelay } = require('./retry.js');
assert.equal(retryDelay(0, 100, 500), 100);
assert.equal(retryDelay(4, 100, 500), 500);
