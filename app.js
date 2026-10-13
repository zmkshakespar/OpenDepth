(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const config = window.OPENDEPTH_CONFIG || {};
  for (const [id, url] of [['arxiv-link', config.arxivUrl], ['code-link', config.codeUrl]]) {
    if (url && /^https:\/\//i.test(url)) {
      const old = document.getElementById(id), a = document.createElement('a');
      a.id = id; a.className = old.className; a.href = url; a.target = '_blank'; a.rel = 'noopener';
      a.textContent = id === 'arxiv-link' ? 'arXiv ↗' : 'Code ↗'; old.replaceWith(a);
    }
  }
  $('#abstract-toggle').addEventListener('click', () => {
    const open = $('#abstract').hidden;
    $('#abstract').hidden = !open;
    $('#abstract-toggle').setAttribute('aria-expanded', String(open));
    $('#abstract-toggle').innerHTML = `${open ? 'Close the abstract' : 'Read the full abstract'} <span>${open ? '−' : '+'}</span>`;
  });
  const setActive = (selector, attr, value) => $$(selector).forEach(b => {
    const active = b.dataset[attr] === value;
    b.classList.toggle('active', active); b.setAttribute('aria-pressed', String(active));
  });
  let scene = 'kitti', weather = 'clean', mode = 'split', loadVersion = 0;
  const weatherNames = {clean:'clean',dark:'low light',fog:'fog',snow:'snow',blur:'motion blur',contrast:'low contrast'};
  const pairPath = kind => `assets/epoch016/playground/${scene}_${weather}_${kind}`;
  const slider = $('#compare-slider');
  slider.addEventListener('input', () => $('#comparison').style.setProperty('--split', `${slider.value}%`));
  const loadImage = src => new Promise((resolve,reject) => { const im = new Image(); im.onload = () => resolve(im); im.onerror = reject; im.src = src; });
  function refreshScene() {
    $('#viewer-rgb').src = pairPath('rgb.jpg'); $('#viewer-depth').src = pairPath('depth.jpg');
    $('#viewer-rgb').alt = `${scene === 'kitti' ? 'KITTI outdoor' : 'NYUv2 indoor'} scene, ${weatherNames[weather]} input`;
    $('#viewer-depth').alt = `OpenDepth relative depth prediction for ${weatherNames[weather]} ${scene} input`;
    $('#comparison').classList.toggle('indoor', scene === 'nyu');
    $('#comparison').style.aspectRatio = scene === 'nyu' ? '1.33' : '3.33';
    $('#scene-note').textContent = `${scene === 'kitti' ? 'KITTI' : 'NYUv2'} · synthetic visual perturbations · epoch016 disparity checkpoint · 10-step predictions`;
    if (mode === 'points') buildCloud();
  }
  $$('[data-scene]').forEach(b => b.addEventListener('click', () => {scene=b.dataset.scene;setActive('[data-scene]','scene',scene);refreshScene();}));
  $$('[data-weather]').forEach(b => b.addEventListener('click', () => {weather=b.dataset.weather;setActive('[data-weather]','weather',weather);refreshScene();}));
  $$('[data-mode]').forEach(b => b.addEventListener('click', () => {
    mode=b.dataset.mode;setActive('[data-mode]','mode',mode);
    $('#comparison').hidden=mode!=='split';$('#point-view').hidden=mode!=='points';$('#point-disclaimer').hidden=mode!=='points';
    if(mode==='points') buildCloud();
  }));
  const canvas=$('#point-canvas'), ctx=canvas.getContext('2d');
  let points=[],yaw=-.13,pitch=.04,drag=null,drawQueued=false;
  const queueDraw=()=>{if(!drawQueued){drawQueued=true;requestAnimationFrame(()=>{drawQueued=false;drawCloud();});}};
  async function buildCloud(){
    const version=++loadVersion;
    try {
      const [rgb,depth]=await Promise.all([loadImage(pairPath('rgb.jpg')),loadImage(pairPath('gray.png'))]);
      if(version!==loadVersion)return;
      const w=200,h=Math.round(w*rgb.height/rgb.width), off=document.createElement('canvas');off.width=w;off.height=h;
      const c=off.getContext('2d',{willReadFrequently:true});c.drawImage(rgb,0,0,w,h);const colors=c.getImageData(0,0,w,h).data;
      c.clearRect(0,0,w,h);c.drawImage(depth,0,0,w,h);const ds=c.getImageData(0,0,w,h).data;
      points=[];
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=(y*w+x)*4;const z=.7+(1-ds[i]/255)*2.5;points.push({x:(x/w-.5)*z*2.5,y:(y/h-.5)*z*2.5*h/w,z:z-1.7,color:`rgb(${colors[i]},${colors[i+1]},${colors[i+2]})`});}
      queueDraw();
    } catch { ctx.fillStyle='#a6e9c9';ctx.fillText('Could not load this prediction. Please try another scene.',20,80); }
  }
  function drawCloud(){
    if(mode!=='points')return;
    const box=canvas.getBoundingClientRect();if(!box.width)return;
    const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
    const w=box.width,h=box.height;ctx.fillStyle='#0b1711';ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='#24392c';ctx.lineWidth=1;
    for(let x=0;x<w;x+=36){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=0;y<h;y+=36){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),scale=Math.min(w*.42,h*1.05);
    const transformed=points.map(p=>{const x=p.x*cy+p.z*sy,z=-p.x*sy+p.z*cy,y=p.y*cx-z*sx,zz=p.y*sx+z*cx;return{x,y,z:zz,color:p.color};}).sort((a,b)=>b.z-a.z);
    for(const p of transformed){const f=scale/(p.z+4.2);const x=w/2+p.x*f,y=h/2+p.y*f;ctx.fillStyle=p.color;const size=Math.max(1.4,2.1*f/100);ctx.fillRect(x,y,size,size);}
  }
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw=Math.max(-.75,Math.min(.75,yaw+(e.clientX-drag.x)*.004));pitch=Math.max(-.45,Math.min(.45,pitch+(e.clientY-drag.y)*.003));drag={x:e.clientX,y:e.clientY};queueDraw();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drag=null);
  $('#reset-view').addEventListener('click',()=>{yaw=-.13;pitch=.04;queueDraw();});
  new ResizeObserver(queueDraw).observe(canvas);
  const galleryNames={nuscenes_night:'nuScenes · Night',robotcar_night:'RobotCar · Night',ds_rain:'DrivingStereo · Rain',ds_fog:'DrivingStereo · Fog'};
  function renderGallery(key){
    $('#gallery-grid').replaceChildren();
    for(let i=0;i<4;i++){
      const fig=document.createElement('figure');fig.className='gallery-item';
      fig.innerHTML=`<div><img src="assets/epoch016/gallery/${key}_${i}_rgb.jpg" alt="${galleryNames[key]} input sample ${i+1}" loading="lazy" width="960" height="520"><img src="assets/epoch016/gallery/${key}_${i}_depth.jpg" alt="OpenDepth relative depth prediction for sample ${i+1}" loading="lazy" width="960" height="520"></div><p><span>${String(i+1).padStart(2,'0')} / ${galleryNames[key].toUpperCase()}</span><span>↗</span></p>`;
      $('#gallery-grid').append(fig);
    }
  }
  $$('[data-gallery]').forEach(b=>b.addEventListener('click',()=>{setActive('[data-gallery]','gallery',b.dataset.gallery);renderGallery(b.dataset.gallery);}));renderGallery('nuscenes_night');
  const suites={
    world:{description:'Real-world nighttime and adverse weather benchmarks · Paper Table 2',labels:['nuScenes-night','RobotCar-night','DS-rain','DS-cloud','DS-fog'],ours:[14.5,17,9.7,12.5,8.5],ac:[19.8,22.7,12.5,14.9,10.3],da:[20,23.9,12.5,15.1,10.3]},
    general:{description:'General-scene zero-shot depth benchmarks · Paper Table 1',labels:['KITTI','NYUv2','Sintel','ETH3D','DIODE (indoor)'],ours:[6.3,3.9,19.9,3.7,5.5],ac:[8.3,5.1,23.5,6.5,7.3],da:[8.3,5.1,24,6.4,7.3]},
    corruption:{description:'Synthetic corruptions on KITTI-C · Paper Table 3',labels:['Dark','Snow','Motion blur','Gaussian noise'],ours:[12.6,11.7,11.8,14.1],ac:[13,11.4,12.6,15.3],da:[13,11.5,12.7,15.7]}
  };
  function renderBenchmarks(key){
    const s=suites[key],max=Math.max(...s.ours,...s.ac,...s.da)*1.12;$('#results-description').textContent=s.description;$('#benchmark-grid').replaceChildren();
    s.labels.forEach((label,i)=>{const el=document.createElement('article');el.className='benchmark';const values=[['ours','OpenDepth',s.ours[i]],['ac','DepthAnything-AC',s.ac[i]],['da','DepthAnything V2',s.da[i]]];
      el.innerHTML=`<h3>${label}</h3>`+values.map(([cls,name,value])=>`<div class="bar-row" aria-label="${name}: AbsRel ${value.toFixed(1)} percent"><div class="bar-track"><div class="bar-fill ${cls}" style="width:${value/max*100}%"></div></div><span>${value.toFixed(1)}</span></div>`).join('')+'<p class="metric-label">ABSREL (%) ↓</p>';$('#benchmark-grid').append(el);
    });
  }
  $$('[data-suite]').forEach(b=>b.addEventListener('click',()=>{setActive('[data-suite]','suite',b.dataset.suite);renderBenchmarks(b.dataset.suite);}));renderBenchmarks('world');
  $('#copy-citation').addEventListener('click',async()=>{
    const value=$('#bibtex').textContent;
    try{await navigator.clipboard.writeText(value);$('#copy-status').textContent='Citation copied.';}
    catch{const r=document.createRange();r.selectNodeContents($('#bibtex'));const s=window.getSelection();s.removeAllRanges();s.addRange(r);$('#copy-status').textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';}
  });
  if('IntersectionObserver' in window&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
    const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');observer.unobserve(e.target);}}),{threshold:.08});
    $$('.reveal').forEach(el=>{el.classList.add('js-reveal');observer.observe(el);});
  }
  window.addEventListener('scroll',()=>{const total=document.documentElement.scrollHeight-innerHeight;$('.scroll-progress').style.width=`${total>0?scrollY/total*100:0}%`;},{passive:true});
})();
