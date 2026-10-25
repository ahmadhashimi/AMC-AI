const $ = id => document.getElementById(id);
const fmt = (n, digits=0) => Number.isFinite(n) ? n.toLocaleString(undefined, {maximumFractionDigits:digits,minimumFractionDigits:digits}) : '—';
const loss = n => fmt(n,4);
const age = seconds => seconds < 60 ? `${Math.floor(seconds)}s ago` : seconds < 3600 ? `${Math.floor(seconds/60)}m ago` : `${Math.floor(seconds/3600)}h ago`;
const duration = s => !Number.isFinite(s) ? 'Unavailable' : `${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`;
let run=null, hoverPoints=[];
const NS='http://www.w3.org/2000/svg';
function svgNode(type, attributes, text) {
  const node=document.createElementNS(NS,type);
  for(const [key,value] of Object.entries(attributes)) node.setAttribute(key,value);
  if(text!==undefined) node.textContent=text;
  $('chart').append(node); return node;
}
function chart() {
  const svg=$('chart'), rect=svg.getBoundingClientRect();
  const w=rect.width, h=rect.height, left=42, right=16, top=18, bottom=36;
  svg.replaceChildren(); hoverPoints=[];
  if(!run || !run.latest || !w) return;
  const events=run.events, latest=run.latest, span=Number($('range').value);
  const minStep=span ? Math.max(0,latest.step-span) : 0;
  const train=events.filter(e=>e.event==='train' && e.step>=minStep && Number.isFinite(e.train_loss));
  const val=events.filter(e=>e.event==='validation' && e.step>=minStep && Number.isFinite(e.validation_loss));
  const series=[];
  if($('show-train').checked) series.push({data:train,key:'train_loss',color:'#245cc7',name:'Training'});
  if($('show-val').checked) series.push({data:val,key:'validation_loss',color:'#087d78',name:'Validation'});
  const all=series.flatMap(s=>s.data.map(e=>e[s.key]));
  $('chart-empty').style.display=all.length?'none':'flex';
  $('chart-empty').textContent=(!series.length)?'Select a loss series above.':'Loss history will appear after the trainer starts recording measurements.';
  if(!all.length) return;
  const low=Math.min(...all), high=Math.max(...all), pad=Math.max(.1,(high-low)*.12);
  const lo=Math.max(0,low-pad), hi=high+pad, end=Math.max(latest.step,minStep+1);
  const x=step=>left+(step-minStep)/(end-minStep)*(w-left-right);
  const y=value=>top+(hi-value)/(hi-lo)*(h-top-bottom);
  for(let i=0;i<=4;i++) {
    const v=lo+(hi-lo)*i/4, py=y(v);
    svgNode('line',{x1:left,y1:py,x2:w-right,y2:py,stroke:'#e5ecf3','stroke-width':1});
    svgNode('text',{x:left-9,y:py+4,'text-anchor':'end',fill:'#536777','font-size':11},v.toFixed(1));
  }
  for(let i=0;i<=4;i++) {
    const step=minStep+(end-minStep)*i/4;
    svgNode('text',{x:x(step),y:h-10,'text-anchor':i===0?'start':i===4?'end':'middle',fill:'#536777','font-size':11},Math.round(step).toLocaleString());
  }
  for(const s of series) {
    const stride=Math.max(1,Math.ceil(s.data.length/1500));
    const points=s.data.filter((_,i)=>i%stride===0 || i===s.data.length-1);
    const d=points.map((e,i)=>`${i?'L':'M'}${x(e.step).toFixed(1)},${y(e[s.key]).toFixed(1)}`).join(' ');
    svgNode('path',{d,fill:'none',stroke:s.color,'stroke-width':s.name==='Validation'?2.7:1.6,'stroke-linejoin':'round','stroke-linecap':'round',opacity:s.name==='Training'?.65:1});
    for(const e of s.data) {
      hoverPoints.push({x:x(e.step),y:y(e[s.key]),step:e.step,value:e[s.key],name:s.name});
      if(s.name==='Validation' || s.data.length===1) svgNode('circle',{cx:x(e.step),cy:y(e[s.key]),r:3.1,fill:s.color});
    }
  }
  $('chart-count').textContent=`${fmt(train.length+val.length)} measurements in range`;
}
function update(data) {
  run=data;
  const health=assessTraining(data);
  $('health').className=`health ${health.level}`;
  $('health-title').textContent=health.title;
  $('health-reason').textContent=health.reason;
  const s=data.latest;
  if(!s) {
    $('connection').textContent='Waiting for training'; $('connection').className='';
    $('notice').textContent='No training measurements or checkpoint found. Start the trainer to begin.';
    return;
  }
  const seconds=Math.max(0,data.server_time-s.time), legacy=data.mode==='checkpoint';
  const stale=!legacy && s.status==='running' && seconds>30;
  const active=!legacy && !stale && s.status==='running';
  $('connection').textContent=legacy?'Checkpoint updates':s.status==='error'?'Training error':s.status==='completed'?'Completed':s.status==='paused'?'Paused':stale?'No recent update':'Live measurements';
  $('connection').className=active?'live':'';
  $('notice').className=`notice${legacy||stale?' warn':''}`;
  $('notice').textContent=legacy?'Your saved checkpoint is visible. For live loss charts, safely stop the trainer, then resume the updated script. Earlier terminal losses were not saved.':stale?'No recent measurements. The trainer may be busy, stopped, or interrupted. Check its terminal; the last checkpoint remains available.':s.status==='paused'?'Training is paused and progress is saved. Resume from the same checkpoint whenever you’re ready.':s.status==='completed'?'The requested dataset passes are complete. Review validation and generated text before another epoch.':'Training is recording measurements. Charts refresh every 2 seconds; validation is sampled at checkpoint intervals.';
  if(s.status==='error'){
    $('notice').className='notice warn';
    $('notice').textContent='The trainer reported an error. Read its terminal traceback and inspect the last saved checkpoint before resuming.';
  }
  const complete=s.epoch_completed>=s.epochs;
  const epochNumber=complete?s.epochs:s.epoch_completed+1;
  const pct=complete?100:Math.min(100,s.next_batch/s.steps_per_epoch*100);
  $('epoch').textContent=`Epoch ${epochNumber} of ${s.epochs}`;
  $('percent').textContent=`${pct.toFixed(2)}%`;
  $('progress').value=pct;
  $('steps').textContent=`${fmt(complete?s.steps_per_epoch:s.next_batch)} / ${fmt(s.steps_per_epoch)} steps this epoch`;
  const speed=active?s.steps_per_second:null;
  $('eta').textContent=complete?'Dataset pass completed':Number.isFinite(speed)&&speed>0?`${duration((s.steps_per_epoch-s.next_batch)/speed)} remaining this epoch`:'Time estimate available with live measurements';
  const lastTrain=[...data.events].reverse().find(e=>e.event==='train');
  const vals=data.events.filter(e=>e.event==='validation');
  const lastVal=vals.at(-1);
  $('train-loss').textContent=loss(lastTrain?.train_loss);
  $('val-loss').textContent=loss(lastVal?.validation_loss);
  $('speed').textContent=fmt(speed,1);
  $('throughput').textContent=fmt(speed ? speed*s.batch_size*s.block_size : null);
  $('saved-step').textContent=s.checkpoint_step!==null&&s.checkpoint_step!==undefined?`Step ${fmt(s.checkpoint_step)}`:'No checkpoint yet';
  $('saved-time').textContent=s.checkpoint_time?`Saved ${age(Math.max(0,data.server_time-s.checkpoint_time))}`:'Waiting for the first save';
  $('unsaved').textContent=legacy?'Current step may be ahead of this save.':s.checkpoint_step!==null?`${fmt(Math.max(0,s.step-s.checkpoint_step))} steps since last save`:`${fmt(s.step)} steps not yet checkpointed`;
  const rows=[['Training tokens',fmt(s.train_tokens)],['Validation tokens',fmt(s.val_tokens)],['Parameters',fmt(s.parameters)],['Device',s.device?.toUpperCase()||'—'],['Batch size',`${s.batch_size} sequences`],['Sequence length',`${s.block_size} tokens`],['Learning rate',String(s.learning_rate)],['Vocabulary',`${fmt(s.tokenizer_vocab)} tokens`],['Training file',`${fmt(s.train_tokens*2/1024**2,1)} MiB`],['Disk space free',`${fmt(data.disk_free_bytes/1024**3,2)} GiB`]];
  $('run-details').replaceChildren(...rows.map(([k,v])=>{
    const div=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');
    dt.textContent=k;dd.textContent=v;div.append(dt,dd);return div;
  }));
  const body=$('event-body');body.replaceChildren();
  const recent=data.events.filter(e=>['train','validation','checkpoint','stopped','completed','resume'].includes(e.event)).slice(-12).reverse();
  for(const e of recent){
    const tr=document.createElement('tr');
    for(const v of [fmt(e.step),{train:'Training',validation:'Validation',checkpoint:'Checkpoint',stopped:'Paused',completed:'Completed',resume:'Resumed'}[e.event],e.event==='validation'?loss(e.validation_loss):e.event==='train'?loss(e.train_loss):'—',new Date(e.time*1000).toLocaleTimeString()]){
      const td=document.createElement('td');td.textContent=v;tr.append(td);
    }body.append(tr);
  }
  if(!recent.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='No loss history recorded yet.';tr.append(td);body.append(tr);}
  if(vals.length>=2){
    const first=vals[0].validation_loss,last=lastVal.validation_loss;
    $('loss-trend').textContent=`Validation ${loss(first)} → ${loss(last)} (${((last-first)/first*100).toFixed(1)}%)`;
  }else $('loss-trend').textContent=vals.length?'One validation measurement; trend not available yet':'Waiting for validation measurements';
  $('updated').textContent=`Updated ${age(seconds)} · Local measurements`;
  chart();
}
async function refresh(){
  try{
    const response=await fetch('/api/run',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    update(await response.json());
  }catch(error){
    $('health').className='health amber';$('health-title').textContent='Dashboard connection unavailable';$('health-reason').textContent='Cannot assess current training. Check the dashboard terminal; this does not mean the model failed.';
    $('connection').textContent='Connection unavailable';$('connection').className='';
    $('notice').className='notice warn';$('notice').textContent='Cannot reach the local dashboard server. Existing measurements remain on screen. Check the dashboard terminal.';
  }finally{setTimeout(refresh,2000);}
}
for(const id of ['range','show-train','show-val'])$(id).addEventListener('change',()=>{$('tooltip').hidden=true;chart();});
new ResizeObserver(chart).observe($('chart-wrap'));
$('chart').addEventListener('pointermove',event=>{
  if(!hoverPoints.length)return;
  const rect=$('chart').getBoundingClientRect(),px=event.clientX-rect.left,py=event.clientY-rect.top;
  const point=hoverPoints.reduce((best,p)=>Math.hypot(p.x-px,p.y-py)<Math.hypot(best.x-px,best.y-py)?p:best);
  const tip=$('tooltip');tip.hidden=false;tip.textContent=`${point.name} loss: ${loss(point.value)}\nStep ${fmt(point.step)}`;
  tip.style.left=`${Math.max(0,Math.min(rect.width-180,point.x+12))}px`;tip.style.top=`${Math.max(0,point.y-66)}px`;
});
$('chart').addEventListener('pointerleave',()=>{$('tooltip').hidden=true;});
refresh();
