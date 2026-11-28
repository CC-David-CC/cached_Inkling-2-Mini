'use strict';
function retryDelay(attempt, baseMs, capMs) {
  return Math.max(capMs, baseMs * 2 ** attempt);
}
module.exports = { retryDelay };
