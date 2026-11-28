const assert = require('node:assert/strict');
const {retryDelay} = require(process.argv[2]);
let count=0, failed=0;
for (const base of [0, 1, 25, 100]) for (const cap of [0, 10, 100, 400, 999]) {
  let expected = Math.min(base, cap);
  for (let attempt=0; attempt<10; attempt++) {
    count++;
    if (retryDelay(attempt,base,cap)!==expected) failed++;
    expected = Math.min(cap, expected + expected);
  }
}
console.log(JSON.stringify({count,failed,passed:count-failed}));
process.exitCode=Number(failed>0);
