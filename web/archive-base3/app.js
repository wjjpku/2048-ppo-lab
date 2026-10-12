'use strict';
const $ = id => document.getElementById(id);
const fmt = x => Number(x).toLocaleString('en-US');
const arrows = ['↑','→','↓','←'], names = ['向上','向右','向下','向左'];
const palette = [null,['#edf0e3','#727d65'],['#e3e7ce','#6e7b5f'],['#d3dcb1','#526548'],['#bdcc91','#4c603e'],['#a9bf74','#425b34'],['#8fab58','#f9fbf1'],['#edd893','#837034'],['#e5c879','#7d632c'],['#d9b861','#fffaf0'],['#ce9c4d','#fffaf0'],['#bd813b','#fffaf0'],['#a76930','#fffaf0']];
let data, meta, originalReplay, selectedCase=null, loadVersion=0, position = 0, running = false, tickTimer = null, motionTimer = null, motionVersion = 0;
const replayCache=new Map();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
function color(el,e){const p=palette[e]||['#655e47','#fff'];el.style.setProperty('--tile-bg',p[0]);el.style.setProperty('--tile-ink',p[1]);}
function currentBoard(){return position ? data.steps[position-1].board_after : data.initial_board;}
function locate(el,r,c){const width=$('board').clientWidth, gap=9, cell=(width-3*gap)/4;el.style.transform=`translate(${c*(cell+gap)}px,${r*(cell+gap)}px)`;}
function makeTile(e,r,c){const el=document.createElement('div');el.className='tile'+(e>=10?' small-number':'');el.textContent=2**e;color(el,e);locate(el,r,c);return el;}
function paint(board, spawn=null, merged=[]){$('tiles').replaceChildren();board.forEach((row,r)=>row.forEach((e,c)=>{if(!e)return;const el=makeTile(e,r,c);if(spawn&&r===spawn.row&&c===spawn.column)el.classList.add('new');if(merged.includes(r*4+c))el.classList.add('merged');$('tiles').append(el);}));$('board').setAttribute('aria-label',`第${position}步棋盘，${board.map(row=>row.map(e=>e?2**e:'空').join('、')).join('；')}`);}
// Build only visual paths from the recorded pre-move board. The final state always comes from the archive.
function paths(board,action){const out=[],merged=[];for(let lane=0;lane<4;lane++){const coords=Array.from({length:4},(_,i)=>action===0?[i,lane]:action===1?[lane,3-i]:action===2?[3-i,lane]:[lane,i]);const tiles=coords.map(([r,c])=>({r,c,e:board[r][c]})).filter(t=>t.e);let dest=0;for(let i=0;i<tiles.length;i++){const [r,c]=coords[dest++];out.push({...tiles[i],toR:r,toC:c});if(i+1<tiles.length&&tiles[i].e===tiles[i+1].e){out.push({...tiles[++i],toR:r,toC:c});merged.push(r*4+c);}}}return {out,merged};}
function animate(step,duration,version){const plan=paths(step.board_before,step.action);$('tiles').replaceChildren();const elements=plan.out.map(t=>{const el=makeTile(t.e,t.r,t.c);$('tiles').append(el);return [el,t];});void $('tiles').offsetWidth;requestAnimationFrame(()=>{if(version!==motionVersion)return;for(const [el,t] of elements){el.style.transition=`transform ${duration}ms cubic-bezier(.25,.65,.35,1)`;locate(el,t.toR,t.toC);}});motionTimer=setTimeout(()=>{if(version===motionVersion)paint(step.board_after,step.spawn,plan.merged);},duration+12);}
function renderInfo(){const s=position?data.steps[position-1]:null,b=currentBoard(),p=s||data.steps[0];$('score').textContent=fmt(s?s.score:0);$('gain').textContent=s?`本步 +${fmt(s.raw_reward)}`:'起始棋盘';$('max-tile').textContent=fmt(2**Math.max(...b.flat()));$('action').textContent=s?arrows[s.action]:'—';$('move-label').textContent=s?names[s.action]:'等待播放';$('step').textContent=position;$('timeline').value=position;$('timeline').setAttribute('aria-valuetext',`第 ${position} 步，共 ${data.steps.length} 步`);$('game-status').textContent=position===data.steps.length?'自然终局':position?`第 ${position} 步`:'初始状态';$('event-step').textContent=position?`第 ${position} 步`:'初始棋盘';$('spawn').textContent=s?`${s.spawn.tile} · 第${s.spawn.row+1}行 第${s.spawn.column+1}列`:'初始 2 块';$('reward').textContent=s?`+${fmt(s.raw_reward)}`:'—';$('empty').textContent=`${b.flat().filter(x=>!x).length} / 16`;$('state').textContent=s?.terminated?'已结束 · 无合法动作':'可继续';$('decision-title').textContent=s?'本步策略':'下一步策略';$('decision-note').textContent=`${arrows[p.action]} ${names[p.action]} · ${(p.chosen_probability*100).toFixed(1)}%`;
 $('probabilities').replaceChildren();p.probabilities.forEach((v,a)=>{const row=document.createElement('div');row.className=`prob-row${a===p.action?' chosen':''}${!p.legal_mask[a]?' illegal':''}`;row.innerHTML=`<span class="prob-name">${arrows[a]} ${names[a].slice(1)}</span><div class="prob-track"><div class="prob-fill" style="width:${v*100}%"></div></div><span class="prob-value">${p.legal_mask[a]?(v*100).toFixed(1)+'%':'不可用'}</span>`;$('probabilities').append(row);});$('first').disabled=$('prev').disabled=position===0;$('next').disabled=$('last').disabled=position===data.steps.length;$('chart-cursor').setAttribute('x1',position/data.steps.length*360);$('chart-cursor').setAttribute('x2',position/data.steps.length*360);updatePlay();}
function updatePlay(){const label=running?'暂停':position===data.steps.length?'重播':'播放';$('play').innerHTML=`${running?'Ⅱ':'▶'} <span>${label}</span>`;$('play').setAttribute('aria-label',running?'暂停回放':position===data.steps.length?'重新播放回放':'播放回放');$('play').setAttribute('aria-pressed',String(running));}
function go(n,animated=false){const old=position;position=Math.max(0,Math.min(data.steps.length,n));clearTimeout(motionTimer);const version=++motionVersion;renderInfo();if(animated&&position===old+1&&!reducedMotion.matches){animate(data.steps[position-1],Math.min(230,480/Number($('speed').value)*.65),version);}else paint(currentBoard(),position?data.steps[position-1].spawn:null);}
function stop(){running=false;clearTimeout(tickTimer);updatePlay();}
function schedule(){clearTimeout(tickTimer);tickTimer=setTimeout(()=>{if(!running)return;go(position+1,true);if(position===data.steps.length)stop();else schedule();},600/Number($('speed').value));}
function play(){if(running){stop();return;}if(position===data.steps.length)go(0);running=true;updatePlay();schedule();}
function jump(n,animated=false){stop();go(n,animated);}
function miniBoard(board,el){el.replaceChildren();board.forEach(row=>row.forEach(e=>{const t=document.createElement('div');t.className='mini-tile'+(!e?' empty':'');if(e){t.textContent=2**e;color(t,e);}el.append(t);}));}
function configureGame(c){
 selectedCase=c;
 $('episode-heading').textContent=`第 ${data.env_id} 号棋局`;
 $('source-kind').textContent=c?`${c.label} · 确定性重建`:'历史原始记录';
 $('verified-badge').textContent=`✓ ${fmt(data.steps.length)} 步已核验`;
 $('episode-selector').value=c?String(c.env_id):'original';
 $('replay-source-note').textContent=c?'原seed + 冻结模型的CPU重建；终局棋盘、分数、步数和最大块均与原记录一致。':'历史保存的完整逐步日志，未经重建。';
 $('total-steps').textContent=data.steps.length;$('timeline').max=data.steps.length;
 $('final-score').textContent=fmt(data.result.score);$('final-length').textContent=`${fmt(data.result.length)} 步`;$('final-max').textContent=`最大块 ${data.result.max_tile}`;
 $('episode-context').textContent=c?`原100局按分数升序第 ${c.rank} 名。原评估成绩保持不变；重建轨迹每步已核验。`:`按分数升序位于100局中的第 ${meta.replay_rank} 名。原评估固定记录第0号棋局，未按得分挑选。`;
 $('verification-link').href=c?'reconstructed/verification.json':'verification.json';
 $('replay-download').href=c?c.replay_url:'replay.json';$('replay-download').download=`base3-4890-bank13-env${data.env_id}${c?'-reconstructed':''}.json`;
 const audit=c?meta.reconstruction.cases.find(x=>x.env_id===c.env_id):null;
 $('replay-hash').textContent=audit?audit.sha256:meta.source_hashes['selected_test_replay.json'];
 for(const el of document.querySelectorAll('.sample')){const active=!!c&&el.dataset.envId===String(c.env_id);el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));}
 $('milestones').replaceChildren();
 for(const tile of [64,128,256,512,1024,2048,4096]){const i=data.steps.findIndex(s=>Math.max(...s.board_after.flat())>=Math.log2(tile));if(i<0)continue;const b=document.createElement('button');b.textContent=`${tile} ↗`;b.title=`首次达到 ${tile}，第 ${i+1} 步`;b.setAttribute('aria-label',b.title);b.onclick=()=>jump(i+1);$('milestones').append(b);}
 const pts=[[0,72],...data.steps.map((s,i)=>[(i+1)/data.steps.length*360,72-s.score/data.result.score*69])];$('score-line').setAttribute('points',pts.map(p=>p.join(',')).join(' '));$('score-area').setAttribute('d','M'+pts.map(p=>p.join(',')).join(' L')+' L360,74 L0,74 Z');
 go(0);
}
async function chooseCase(value,scroll=false){
 stop();const version=++loadVersion;const c=value==='original'?null:meta.cases.find(x=>String(x.env_id)===String(value));
 $('replay-source-note').textContent='正在读取完整轨迹…';
 try{
  let replay=originalReplay;
  if(c){if(!c.has_replay||!meta.reconstruction?.verified)throw new Error('该案例尚无经过核验的完整轨迹');if(!replayCache.has(c.env_id)){const response=await fetch(c.replay_url);if(!response.ok)throw new Error(`HTTP ${response.status}`);replayCache.set(c.env_id,await response.json());}replay=replayCache.get(c.env_id);if(replay.env_id!==c.env_id||replay.steps.length!==c.length||replay.result.score!==c.score||replay.result.max_tile!==c.max_tile||!replay.complete)throw new Error('案例身份与核验清单不匹配');}
  if(version!==loadVersion)return;clearTimeout(motionTimer);motionVersion++;data=replay;configureGame(c);
  if(scroll)document.querySelector('.replay-panel').scrollIntoView({behavior:reducedMotion.matches?'auto':'smooth',block:'start'});
 }catch(error){if(version!==loadVersion)return;$('episode-selector').value=selectedCase?String(selectedCase.env_id):'original';$('replay-source-note').textContent=`加载失败：${error.message}。当前仍为第${data.env_id}号棋局。`;}
}
function loadSamples(){
 $('episode-selector').replaceChildren();const original=document.createElement('option');original.value='original';original.textContent='原始记录 · 第0号局 · 14,408分';$('episode-selector').append(original);
 for(const c of meta.cases){const option=document.createElement('option');option.value=c.env_id;option.textContent=`${c.label} · 第${c.env_id}号局 · ${fmt(c.score)}分`; $('episode-selector').append(option);
  const el=document.createElement('button');el.className='sample';el.dataset.envId=c.env_id;el.setAttribute('aria-label',`播放${c.label}完整回放，${fmt(c.score)}分，第${c.env_id}号棋局`);el.innerHTML=`<span class="sample-label">${c.label}<span>#${c.rank} / 100</span></span><strong class="sample-score">${fmt(c.score)}</strong><div class="mini-board" aria-hidden="true"></div><div class="sample-bottom"><span>最大 ${c.max_tile}</span><span>${fmt(c.length)} 步</span></div><span class="sample-view">▶ 播放全部 ${fmt(c.length)} 步</span>`;miniBoard(c.final_board,el.querySelector('.mini-board'));el.onclick=()=>chooseCase(c.env_id,true);$('samples').append(el);
 }
 $('episode-selector').onchange=e=>chooseCase(e.target.value);
}
function setup(){for(let i=0;i<16;i++)document.querySelector('.grid-bg').append(document.createElement('i'));$('first').onclick=()=>jump(0);$('prev').onclick=()=>jump(position-1);$('next').onclick=()=>jump(position+1,true);$('last').onclick=()=>jump(data.steps.length);$('play').onclick=play;$('timeline').oninput=e=>jump(Number(e.target.value));$('speed').onchange=()=>{if(running)schedule();};$('close-dialog').onclick=()=>$('snapshot-dialog').close();$('snapshot-dialog').addEventListener('click',e=>{if(e.target===$('snapshot-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
 document.addEventListener('keydown',e=>{if($('snapshot-dialog').open||e.target.closest('input,select,button,a,summary')||e.target.isContentEditable)return;if([' ','ArrowRight','ArrowLeft','Home','End'].includes(e.key))e.preventDefault();if(e.key===' ')play();if(e.key==='ArrowRight')jump(position+1,true);if(e.key==='ArrowLeft')jump(position-1);if(e.key==='Home')jump(0);if(e.key==='End')jump(data.steps.length);});new ResizeObserver(()=>{clearTimeout(motionTimer);motionVersion++;paint(currentBoard(),position?data.steps[position-1].spawn:null);}).observe($('board'));
 $('model-hash').textContent=meta.checkpoint_sha256;$('selection-rule').textContent=meta.selection_rule;loadSamples();configureGame(null);}
Promise.all(['replay.json','metadata.json'].map(async url=>{const r=await fetch(url);if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return r.json();})).then(([replay,metadata])=>{data=replay;originalReplay=replay;meta=metadata;if(!meta.verification.valid||data.steps.length!==835||data.result.score!==14408)throw new Error('回放数据未通过来源检查');$('loading').hidden=true;$('app').hidden=false;setup();}).catch(error=>{document.body.classList.add('failed');$('loading').textContent=`无法加载真实回放：${error.message}。请检查本地数据文件。`;console.error(error);});
