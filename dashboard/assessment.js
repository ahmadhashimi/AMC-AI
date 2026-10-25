/* Monitoring signals, not a model-quality or factual-accuracy assessment. */
(function (root) {
  function assess(run) {
    const s=run?.latest;
    if(!s)return {level:'amber',title:'Waiting for training',reason:'No measurements are available yet.'};
    if(s.status==='error')return {level:'red',title:'Training error — check the terminal',reason:s.error || 'Training reported an error. Review it before resuming.'};
    const vals=(run.events||[]).filter(e=>e.event==='validation' && Number.isFinite(e.validation_loss));
    if((run.events||[]).some(e=>['train','validation'].includes(e.event) && ((e.train_loss!==null && e.train_loss!==undefined && !Number.isFinite(e.train_loss)) || (e.validation_loss!==null && e.validation_loss!==undefined && !Number.isFinite(e.validation_loss))))) {
      return {level:'red',title:'Invalid loss measurement',reason:'A loss is not finite. Check the training terminal before continuing.'};
    }
    if(s.status==='paused')return {level:'amber',title:'Training paused — progress saved',reason:'Resume when ready. A pause is not a learning failure.'};
    if(run.mode==='checkpoint')return {level:'amber',title:'Live losses unavailable',reason:'Only a saved checkpoint is visible. Resume the updated trainer to record loss history.'};
    if(s.status==='running' && run.server_time-s.time>30)return {level:'amber',title:'No recent measurements',reason:'Check the training terminal. This can mean a slow step, interruption, or stopped process.'};
    if(vals.length<2)return {level:'amber',title:'Gathering validation measurements',reason:'Wait for at least two validation checkpoints before judging the trend.'};
    const recent=vals.slice(-4),first=recent[0].validation_loss,last=recent.at(-1).validation_loss;
    const change=(last-first)/first;
    const rising=recent.length===4 && recent.slice(1).every((e,i)=>e.validation_loss>recent[i].validation_loss);
    if(rising && change>=.02)return {level:'red',title:'Validation keeps worsening — review training',reason:`Validation loss rose at three consecutive checks, up ${(change*100).toFixed(1)}% over these checks. Possible overfitting or instability; inspect before epoch 2.`};
    if(change>=.005)return {level:'amber',title:'Validation is higher — watch the next checks',reason:`Recent validation loss increased ${(change*100).toFixed(1)}%. One noisy interval is not proof of failure.`};
    if(change<=-.005)return {level:'green',title:s.status==='completed'?'Pass complete — validation improved':vals.length<4?'Early signs of learning':'Training is improving',reason:`Recent sampled validation loss fell ${(-change*100).toFixed(1)}% (${first.toFixed(4)} to ${last.toFixed(4)}). Keep monitoring; this does not establish answer quality.`};
    return {level:'amber',title:'Validation is steady or mixed',reason:'Recent change is under 0.5%. It may be noise or a plateau. Review more checkpoints and fixed-prompt output.'};
  }
  if(typeof module!=='undefined' && module.exports)module.exports=assess;
  else root.assessTraining=assess;
})(typeof globalThis!=='undefined'?globalThis:this);
