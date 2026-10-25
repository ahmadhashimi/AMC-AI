const assert = require('node:assert/strict');
const assess = require('../dashboard/assessment.js');
function run(losses, status='running') {
  return {mode:'metrics', server_time:100, latest:{status,time:100}, events:losses.map((validation_loss,i)=>({event:'validation',step:(i+1)*1000,validation_loss}))};
}
assert.equal(assess(run([4.8302,4.7921])).level,'green');
assert.equal(assess(run([5.0])).level,'amber');
assert.equal(assess(run([5,5.04,5.08,5.12])).level,'red');
assert.equal(assess(run([5,5.1,4.99,5.02])).level,'amber');
assert.equal(assess(run([5,4.8],'paused')).level,'amber');
assert.equal(assess(run([5,4.8],'error')).level,'red');
const stale=run([5,4.8]);stale.server_time=200;
assert.equal(assess(stale).level,'amber');
const legacy=run([]);legacy.mode='checkpoint';
assert.equal(assess(legacy).level,'amber');
console.log('PASS: improvement, insufficient history, sustained worsening, mixed, paused, error, stale and legacy states.');
