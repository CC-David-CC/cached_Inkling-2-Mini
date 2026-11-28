'use strict';
function retryDelay(attempt, baseMs, capMs) {
  return Math.min(capMs, baseMs * 2 ** attempt);
}
module.exports = { retryDelay };
