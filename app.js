/* ============================================================
   GouTu3 算法（严格对齐 notebook，不新增）
   A 传统色块 → B mock AI → C 色相差冲突 → D finalize → E evaluate
   ============================================================ */
const PHI = 1.618;

function classifyHue(h){
  const d=h*2;
  if(d<15||d>=345) return "Red";
  if(d<45)  return "Orange/Yellow";
  if(d<165) return "Green";
  if(d<255) return "Blue";
  return "Purple";
}

/* A: 像素级 HSV 色块统计 */
function traditionalBlocks(imgEl, W, H){
  const off=document.createElement('canvas'); off.width=W; off.height=H;
  const oc=off.getContext('2d'); oc.drawImage(imgEl,0,0,W,H);
  const data=oc.getImageData(0,0,W,H).data;

  function rgbToHsv(r,g,b){
    r/=255;g/=255;b/=255;
    const mx=Math.max(r,g,b),mn=Math.min(r,g,b),d=mx-mn;
    let h=0,s=mx===0?0:d/mx,v=mx;
    if(d>0){
      if(mx===r)      h=((g-b)/d)%6;
      else if(mx===g) h=(b-r)/d+2;
      else            h=(r-g)/d+4;
      h=Math.round(h*30); if(h<0)h+=180;
    }
    return [h,Math.round(s*255),Math.round(v*255)];
  }

  const groups={};
  for(let y=0;y<H;y++) for(let x=0;x<W;x++){
    const i=(y*W+x)*4;
    const [h,s,v]=rgbToHsv(data[i],data[i+1],data[i+2]);
    if(s<30) continue;
    const cat=classifyHue(h);
    if(!groups[cat]) groups[cat]={cat,hSum:0,sSum:0,vSum:0,cnt:0,
      minX:W,maxX:0,minY:H,maxY:0};
    const g=groups[cat];
    g.hSum+=h;g.sSum+=s;g.vSum+=v;g.cnt++;
    if(x<g.minX)g.minX=x;if(x>g.maxX)g.maxX=x;
    if(y<g.minY)g.minY=y;if(y>g.maxY)g.maxY=y;
  }

  const blocks=[];
  Object.values(groups).forEach(g=>{
    if(g.cnt<100) return;
    const n=g.cnt;
    const mh=g.hSum/n,ms=g.sSum/n,mv=g.vSum/n;
    const cx=(g.minX+g.maxX)/2,cy=(g.minY+g.maxY)/2;
    blocks.push({
      color:g.cat, hue:mh, sat:ms/255, val:mv/255, area:n,
      center:[cx/W,cy/H],
      bbox:[g.minX/W,g.minY/H,(g.maxX-g.minX)/W,(g.maxY-g.minY)/H],
    });
  });
  return blocks;
}

/* B: mock AI regions */
function mockAiRegions(blocks){
  if(!blocks.length) return [];
  const sorted=[...blocks].sort((a,b)=>(b.sat-a.sat)*0.6+(b.area-a.area)/1e5*0.4);
  const bird=sorted[0];
  return [{class_name:'bird',hsv_mean:[bird.hue,bird.sat*255,bird.val*255],area:bird.area}];
}

/* C: hue_difference */
function hueDiff(h1,h2){ const d=Math.abs(h1-h2); return Math.min(d,180-d); }
function detectConflicts(blocks,aiR,thr=10){
  const out=[];
  blocks.forEach((b,ti)=>{
    aiR.forEach(ar=>{
      const d=hueDiff(b.hue,ar.hsv_mean[0]);
      if(d>thr) out.push({ti,blockColor:b.color,aiClass:ar.class_name,diff:d.toFixed(1)});
    });
  });
  return out;
}

/* D: finalize（透传） */
function finalizeBlocks(b){ return b; }

/* E: compute_score */
// 取 top-3 色块作锚点——只给构图抽屉画色块/锚点图层用的可视化辅助，
// 和评分本身无关，评分已经换成后端 bird_classification 管线的结果。
function pickAnchors(blocks){
  if(!blocks.length) return [];
  const sorted=[...blocks].sort((a,b)=>{
    if(Math.abs(b.sat-a.sat)>0.01) return b.sat-a.sat;
    if(Math.abs(a.val-b.val)>0.01) return a.val-b.val;
    return b.area-a.area;
  });
  return sorted.slice(0,3);
}

// 评分来源：后端 /analyze 返回的 aesthetic{composition,richness,purity} 缓存在
// m._aesthetic 上（只在分析成功时写一次），这里只是拿缓存的三个维度分做一次
// 加权求和——纯算术，不发请求，权重滑杆拖动可以保持实时响应。
// 对外字段名沿用 pos/ratio/cons/total/anchors，兼容偏好学习系统等既有读取点。
function deriveScore(m){
  const a=m._aesthetic;
  if(!a) return null;
  const w=(typeof S!=='undefined'&&S.weights)||{pos:0.3,ratio:0.4,cons:0.3};
  return {
    pos:Math.round(a.composition), ratio:Math.round(a.richness), cons:Math.round(a.purity),
    anchors:pickAnchors(m.blocks),
    total:Math.round(w.pos*a.composition + w.ratio*a.richness + w.cons*a.purity),
  };
}

/* 构图方案位置 */
function buildSchemes(){
  const cx=0.5,cy=0.5,diagN=Math.hypot(0.5,0.5);
  const tPts=[[1/3,1/3],[2/3,1/3],[1/3,2/3],[2/3,2/3]];
  let tB=tPts[0],tSc=-1;
  tPts.forEach(p=>{const s=Math.hypot(p[0]-cx,p[1]-cy)/diagN;if(s>tSc){tSc=s;tB=p;}});
  const gx=1/PHI,gy=1/PHI;
  const gPts=[[gx,gy],[1-gx,gy],[gx,1-gy],[1-gx,1-gy]];
  let gB=gPts[0],gSc=-1;
  gPts.forEach(p=>{const mx=Math.abs(p[0]-cx)/.5,my=Math.abs(p[1]-cy)/.5;
    const s=(mx+my)/2;if(s>gSc){gSc=s;gB=p;}});
  const sOpts=[{p:[.5,1/3],t:"vertical",d:"垂直对称"},{p:[1/3,.5],t:"horizontal",d:"水平对称"},{p:[.5,.5],t:"central",d:"中心对称"}];
  let sB=sOpts[0],sSc=-1;
  sOpts.forEach(o=>{let s=1-Math.hypot(o.p[0]-cx,o.p[1]-cy)/diagN;if(o.t==="central")s*=1.2;if(s>sSc){sSc=s;sB=o;}});
  const rs=[0.3,0.4,0.6,0.7];let dB=[0.3,0.3],dSc=-1;
  rs.forEach(r=>{const x=r,y=r;const e=Math.min(x,1-x,y,1-y)/.5;const dy=Math.abs(x-y);
    const s=e*.6+dy*.4;if(s>dSc){dSc=s;dB=[x,y];}});
  const C=(v)=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  return {
    thirds:  {name:"三分法",  color:C('--c-thirds'), score:tSc, pos:tB},
    golden:  {name:"黄金分割",color:C('--c-golden'), score:gSc, pos:gB},
    symmetry:{name:"对称",    color:C('--c-sym'),    score:sSc, pos:sB.p, sym:sB.t},
    diagonal:{name:"对角线",  color:C('--c-diag'),   score:dSc, pos:dB},
  };
}

/* ============================================================
   偏好学习（喜欢/不喜欢）
   独立于 GouTu3 算法，只读取 computeScore/buildSchemes 的输出，
   不修改、不参与它们的计算。
   ============================================================ */
const PREF_KEY='neatpic_prefs_v1';
const MIN_SAMPLES=3;
function loadPrefs(){
  try{ return JSON.parse(localStorage.getItem(PREF_KEY)||'[]'); }
  catch(e){ return []; }
}
function savePrefs(){
  try{ localStorage.setItem(PREF_KEY, JSON.stringify(PREFS)); }
  catch(e){}
}
let PREFS = loadPrefs();

function pushPrefSample(m, liked){
  PREFS.push({
    scheme:DS.scheme, subjectRatio:DS.ratio,
    pos:m.score.pos, colorRatio:m.score.ratio, cons:m.score.cons, total:m.score.total,
    liked, ts:Date.now(),
  });
  savePrefs();
}

function computePreferenceProfile(){
  if(PREFS.length < MIN_SAMPLES) return null;
  const liked=PREFS.filter(p=>p.liked);
  const schemeAffinity={};
  ['thirds','golden','symmetry','diagonal'].forEach(k=>{
    schemeAffinity[k]=PREFS.filter(p=>p.scheme===k&&p.liked).length - PREFS.filter(p=>p.scheme===k&&!p.liked).length;
  });
  function avgLiked(key){ return liked.length ? liked.reduce((s,p)=>s+p[key],0)/liked.length : null; }
  return {
    schemeAffinity,
    idealSubjectRatio:avgLiked('subjectRatio'), idealPos:avgLiked('pos'),
    idealColorRatio:avgLiked('colorRatio'), idealCons:avgLiked('cons'),
    sampleCount:PREFS.length, likedCount:liked.length,
  };
}

function recommendedScheme(profile){
  if(!profile) return null;
  const entries=Object.entries(profile.schemeAffinity).sort((a,b)=>b[1]-a[1]);
  return entries[0][1]>0 ? entries[0][0] : null;
}

function computePreferenceMatch(m, scheme, subjectRatio, profile){
  if(!profile || !m.score) return null;
  const aff=Object.values(profile.schemeAffinity);
  const lo=Math.min(...aff), hi=Math.max(...aff), range=hi-lo;
  const schemeComp=range===0 ? 50 : ((profile.schemeAffinity[scheme]-lo)/range)*100;
  function closeness(val, ideal, scale){ return ideal===null ? 50 : Math.max(0,100-Math.abs(val-ideal)/scale*100); }
  const ratioComp=closeness(subjectRatio, profile.idealSubjectRatio, 15);
  const posComp=closeness(m.score.pos, profile.idealPos, 60);
  const colorComp=closeness(m.score.ratio, profile.idealColorRatio, 60);
  const consComp=closeness(m.score.cons, profile.idealCons, 60);
  return Math.round(schemeComp*0.4 + ratioComp*0.15 + posComp*0.15 + colorComp*0.15 + consComp*0.15);
}

// 图库里每张照片对当前偏好画像的匹配情况，按匹配度（无画像时按综合分）降序排列
function computeGalleryOverview(profile){
  return GALLERY.map(m=>{
    const scheme=m.scheme||'thirds';
    const subjectRatio=m.ratio||20;
    const match=computePreferenceMatch(m, scheme, subjectRatio, profile);
    return {m, scheme, subjectRatio, match, total:m.score?m.score.total:0};
  }).sort((a,b)=> profile ? (b.match-a.match) : (b.total-a.total));
}

/* ============================================================
   图库状态
   ============================================================ */
// 每张图: {id, name, dataURL, el(Image), blocks, score, scheme, ratio}
const GALLERY = [];
let galNextId = 1;

function addImages(files){
  const proms=[];
  for(const f of files){
    if(!f.type.startsWith('image/')) continue;
    proms.push(new Promise(res=>{
      const reader=new FileReader();
      reader.onload=e=>{
        const dataURL=e.target.result;
        const img=new Image();
        img.onload=()=>{
          const id='img'+(galNextId++);
          const blocks=traditionalBlocks(img,img.naturalWidth,img.naturalHeight);
          const aiR=mockAiRegions(blocks);
          const conflicts=detectConflicts(blocks,aiR,10);
          // 评分不再上传时同步算：交给 bird_classification 后端的 /analyze，
          // 结果缓存到 _aesthetic，上传后台自动触发分析（见 handleFiles）。
          GALLERY.push({id,name:f.name,dataURL,el:img,blocks,conflicts,
            conflictChoice:{},conflictIdx:0,score:null,_aesthetic:null,scheme:'thirds',ratio:20,liked:null,recognize:null,
            edit:{brightness:100,contrast:100,saturate:100,hue:0}});
          res();
        };
        img.src=dataURL;
      };
      reader.readAsDataURL(f);
    }));
  }
  return Promise.all(proms);
}

/* ============================================================
   图库渲染
   ============================================================ */
function matchBadgeHtml(m, profile){
  const match=computePreferenceMatch(m, m.scheme||'thirds', m.ratio||20, profile);
  return match===null ? '' : `<span class="gc-match">♥ ${match}</span>`;
}

function renderGallery(filter){
  const grid=document.getElementById('galGrid');
  const q=(filter||'').toLowerCase();
  const list=q?GALLERY.filter(m=>m.name.toLowerCase().includes(q)):GALLERY;
  document.getElementById('galCount').textContent=list.length+' 张';
  if(!list.length){
    grid.innerHTML='<div class="gal-empty" id="galEmpty">还没有图片，点击上方「上传图片」或拖拽到上面添加</div>';
    return;
  }
  const profile=computePreferenceProfile();
  grid.innerHTML=list.map(m=>`
    <div class="gal-card" data-id="${m.id}">
      <img class="gc-img" src="${m.dataURL}" alt="${m.name}" loading="lazy">
      <div class="gc-body">
        <div class="gc-name" title="${m.name}">${m.name}</div>
        <div class="gc-meta">${m.score?m.score.total+' 分':'—'}</div>
      </div>
      <span class="gc-score">${m.score?m.score.total:'—'}</span>
      ${matchBadgeHtml(m, profile)}
      <div class="gc-actions">
        <div class="gac-btn compose" data-id="${m.id}"><svg class="ic"><use href="#ic-compose"/></svg>调试构图</div>
        <div class="gac-btn detail" data-detail="${m.id}" title="详情 / 调色"><svg class="ic"><use href="#ic-target"/></svg></div>
        <div class="gac-btn" data-del="${m.id}"><svg class="ic"><use href="#ic-trash"/></svg></div>
      </div>
    </div>
  `).join('');

  // 绑定点击
  grid.querySelectorAll('.gac-btn.compose').forEach(btn=>{
    btn.onclick=e=>{e.stopPropagation();openDrawer(btn.dataset.id);};
  });
  grid.querySelectorAll('[data-detail]').forEach(btn=>{
    btn.onclick=e=>{e.stopPropagation();const m=GALLERY.find(x=>x.id===btn.dataset.detail);if(m)openReportDrawer(m);};
  });
  grid.querySelectorAll('[data-del]').forEach(btn=>{
    btn.onclick=e=>{e.stopPropagation();deleteImg(btn.dataset.del);};
  });
}

function deleteImg(id){
  const idx=GALLERY.findIndex(m=>m.id===id);
  if(idx>=0){GALLERY.splice(idx,1);renderGallery();}
  if(S.currentId===id){S.currentId=null;syncComposePage();}
}

/* ============================================================
   构图抽屉
   ============================================================ */
const DCV=document.getElementById('drawerCvs');
const DCTX=DCV.getContext('2d');
const DW=DCV.width,DH=DCV.height;

const DS={
  imgId:null, scheme:'thirds', ratio:20,
  layers:{lines:true,anchor:true,blocks:false,grid:false},
};

function openDrawer(id){
  const m=GALLERY.find(x=>x.id===id);
  if(!m) return;
  DS.imgId=id; DS.scheme=m.scheme||'thirds'; DS.ratio=m.ratio||20;
  document.getElementById('drawerTitle').textContent='调试构图 · '+m.name;
  document.getElementById('dRatio').value=DS.ratio;
  document.getElementById('dRatioD').textContent='1/'+(100/DS.ratio).toFixed(1);
  buildDrawerSchemes();
  document.getElementById('composeDrawer').classList.add('open');
  updateRateButtons(m);
  drawerRender();
}
function closeDrawer(){ document.getElementById('composeDrawer').classList.remove('open'); }

function updateRateButtons(m){
  document.getElementById('rateLike').classList.toggle('active', m.liked===true);
  document.getElementById('rateDislike').classList.toggle('active', m.liked===false);
}

function buildDrawerSchemes(){
  const schemes=buildSchemes();
  const profile=computePreferenceProfile();
  const rec=recommendedScheme(profile);
  const order=['thirds','golden','symmetry','diagonal'];
  document.getElementById('drawerSchemes').innerHTML=order.map(k=>{
    const s=schemes[k];
    return `<span class="scheme-pill${k===DS.scheme?' sel':''}${k===rec?' recommended':''}"
      data-k="${k}" style="${k===DS.scheme?'background:'+s.color+';':''}"
    >${k===rec?'★ ':''}${s.name} ${s.score.toFixed(2)}</span>`;
  }).join('');
  document.querySelectorAll('.scheme-pill').forEach(el=>{
    el.onclick=()=>{DS.scheme=el.dataset.k;buildDrawerSchemes();drawerRender();};
  });
}

function drawerRender(){
  const m=GALLERY.find(x=>x.id===DS.imgId);
  if(!m) return;
  const schemes=buildSchemes();
  const sc=schemes[DS.scheme];
  DCTX.clearRect(0,0,DW,DH);
  drawCoverCtx(DCTX,m.el,DW,DH);
  if(DS.layers.blocks) drawBlocksCtx(DCTX,m.blocks,DW,DH);
  if(DS.layers.bird||true){
    const [px,py]=sc.pos;
    const tArea=(DS.ratio/100)*DW*DH;
    const ar=m.el.naturalWidth/m.el.naturalHeight;
    const bh=Math.sqrt(tArea/ar),bw=bh*ar;
    const x=px*DW-bw/2,y=py*DH-bh/2;
    DCTX.save();
    DCTX.shadowColor='rgba(0,0,0,.22)';DCTX.shadowBlur=16;DCTX.shadowOffsetY=5;
    roundClipCtx(DCTX,m.el,x,y,bw,bh,10,DW,DH);
    DCTX.restore();
  }
  if(DS.layers.grid){
    const mg=Math.min(DW,DH)*.1;
    DCTX.strokeStyle='rgba(255,255,255,.28)';DCTX.lineWidth=1;
    DCTX.strokeRect(mg,mg,DW-2*mg,DH-2*mg);
  }
  if(DS.layers.lines) drawLinesCtx(DCTX,sc,DS.scheme,DW,DH);
  if(DS.layers.anchor && m.score && m.score.anchors){
    m.score.anchors.forEach((b,i)=>{
      const [cx,cy]=b.center;
      DCTX.fillStyle=sc.color;
      DCTX.beginPath();DCTX.arc(cx*DW,cy*DH,i===0?8:6,0,7);DCTX.fill();
      DCTX.strokeStyle='#fff';DCTX.lineWidth=2;DCTX.stroke();
    });
  }
  // 评分 kv
  const sc2=m.score;
  const profile=computePreferenceProfile();
  const match=computePreferenceMatch(m, DS.scheme, DS.ratio, profile);
  const matchLine=match!==null
    ? `<br>偏好匹配 <b style="color:var(--accent-2)">${match}</b> / 100（基于 ${profile.sampleCount} 次评价）`
    : `<br><span style="color:var(--ink-3)">偏好学习中…（还需 ${MIN_SAMPLES-PREFS.length} 次评价即可解锁匹配度）</span>`;
  document.getElementById('drawerScoreKV').innerHTML=sc2?
    `综合 <b>${sc2.total}</b> · 锚点 <b>${sc2.pos}</b> · 比例 <b>${sc2.ratio}</b> · 一致性 <b>${sc2.cons}</b><br>方案 <b>${schemes[DS.scheme].name}</b> (${schemes[DS.scheme].score.toFixed(2)})${matchLine}`:
    '—';
}

/* ============================================================
   Canvas 工具函数
   ============================================================ */
function drawCoverCtx(c,img,W,H){
  const ar=img.naturalWidth/img.naturalHeight,cr=W/H;
  let dw,dh,dx,dy;
  if(ar>cr){dh=H;dw=H*ar;dx=(W-dw)/2;dy=0;}
  else{dw=W;dh=W/ar;dx=0;dy=(H-dh)/2;}
  c.drawImage(img,dx,dy,dw,dh);
}
function roundClipCtx(c,img,x,y,w,h,r,W,H){
  c.save();
  c.beginPath();
  c.moveTo(x+r,y);c.arcTo(x+w,y,x+w,y+h,r);c.arcTo(x+w,y+h,x,y+h,r);
  c.arcTo(x,y+h,x,y,r);c.arcTo(x,y,x+w,y,r);c.closePath();c.clip();
  const ar=img.naturalWidth/img.naturalHeight,tr=w/h;
  let dw,dh,dx,dy;
  if(ar>tr){dh=h;dw=h*ar;dx=x-(dw-w)/2;dy=y;}
  else{dw=w;dh=w/ar;dx=x;dy=y-(dh-h)/2;}
  c.drawImage(img,dx,dy,dw,dh);c.restore();
}
const BLOCK_CLR={"Red":"rgba(255,77,77,.32)","Orange/Yellow":"rgba(255,165,0,.32)",
  "Green":"rgba(60,179,113,.32)","Blue":"rgba(65,105,225,.32)","Purple":"rgba(147,112,219,.32)"};
function drawBlocksCtx(c,blocks,W,H){
  blocks.forEach(b=>{
    const [bx,by,bw,bh]=b.bbox;
    c.fillStyle=BLOCK_CLR[b.color]||'rgba(200,200,200,.25)';
    c.fillRect(bx*W,by*H,bw*W,bh*H);
  });
}
function drawLinesCtx(c,sc,scheme,W,H){
  c.globalAlpha=.85;c.strokeStyle=sc.color;c.fillStyle=sc.color;c.lineWidth=2;
  function ln(x1,y1,x2,y2,lw){c.save();if(lw)c.lineWidth=lw;c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();c.restore();}
  function dt(x,y,r){c.beginPath();c.arc(x,y,r,0,7);c.fill();c.strokeStyle='rgba(255,255,255,.8)';c.lineWidth=1.5;c.stroke();}
  if(scheme==='thirds'){
    [1/3,2/3].forEach(x=>ln(x*W,0,x*W,H));
    [1/3,2/3].forEach(y=>ln(0,y*H,W,y*H));
    [[1/3,1/3],[2/3,1/3],[1/3,2/3],[2/3,2/3]].forEach(p=>dt(p[0]*W,p[1]*H,4));
  } else if(scheme==='golden'){
    const gx=W/PHI,gy=H/PHI;
    [gx,W-gx].forEach(x=>ln(x,0,x,H));[gy,H-gy].forEach(y=>ln(0,y,W,y));
    [[gx,gy],[W-gx,gy],[gx,H-gy],[W-gx,H-gy]].forEach(p=>dt(p[0],p[1],5));
  } else if(scheme==='symmetry'){
    if(sc.sym==='vertical')ln(W/2,0,W/2,H,3);
    else if(sc.sym==='horizontal')ln(0,H/2,W,H/2,3);
    else{ln(W/2,0,W/2,H);ln(0,H/2,W,H/2);}
  } else if(scheme==='diagonal'){
    ln(0,0,W,H);ln(W,0,0,H);dt(sc.pos[0]*W,sc.pos[1]*H,6);
  }
  c.globalAlpha=1;
}

/* ============================================================
   构图台主状态
   ============================================================ */
const S={
  currentId:null,
  weights:{pos:.3,ratio:.4,cons:.3},
  conflictIdx:0,
};

function syncComposePage(){
  const m=S.currentId?GALLERY.find(x=>x.id===S.currentId):GALLERY[0];
  if(!m || !GALLERY.length){
    document.getElementById('composeNoImg').style.display='';
    document.getElementById('composeHasImg').classList.add('hidden');
    return;
  }
  S.currentId=m.id;
  document.getElementById('composeNoImg').style.display='none';
  document.getElementById('composeHasImg').classList.remove('hidden');
  // 用缓存的后端美学分（m._aesthetic）+ 当前权重重算加权总分，纯算术不发请求
  const sc=deriveScore(m);
  m.score=sc;
  const schemes=buildSchemes();
  const circ=2*Math.PI*28;
  const heroMatchEl=document.getElementById('heroMatch');
  if(!sc){
    // 还没跑过识别台的 /analyze，评分暂缺
    document.getElementById('ringArc').style.strokeDashoffset=circ;
    document.getElementById('heroNum').textContent='—';
    document.getElementById('heroScheme').textContent='方案：'+(schemes[m.scheme||'thirds'].name)+' · 分数 '+schemes[m.scheme||'thirds'].score.toFixed(2);
    document.getElementById('heroImg').textContent='图片：'+m.name;
    heroMatchEl.textContent='评分中…（去识别台跑一次分析即可拿到美学评分）';
    document.getElementById('scPos').textContent='—'; document.getElementById('scPosBar').style.width='0%';
    document.getElementById('scRatio').textContent='—'; document.getElementById('scRatioBar').style.width='0%';
    document.getElementById('scCons').textContent='—'; document.getElementById('scConsBar').style.width='0%';
    updateConflictBanner(m);
    renderComposeSidePanel(computePreferenceProfile());
    return;
  }
  // 更新 UI
  document.getElementById('ringArc').style.strokeDashoffset=circ*(1-sc.total/100);
  document.getElementById('heroNum').textContent=sc.total;
  document.getElementById('heroScheme').textContent='方案：'+(schemes[m.scheme||'thirds'].name)+' · 分数 '+schemes[m.scheme||'thirds'].score.toFixed(2);
  document.getElementById('heroImg').textContent='图片：'+m.name;
  const profile=computePreferenceProfile();
  const match=computePreferenceMatch(m, m.scheme||'thirds', m.ratio||20, profile);
  const rec=recommendedScheme(profile);
  if(match===null){
    heroMatchEl.textContent='偏好学习中…（还需 '+(MIN_SAMPLES-PREFS.length)+' 次评价即可解锁匹配度）';
  }else{
    let txt='偏好匹配 '+match+' / 100';
    if(rec && rec!==(m.scheme||'thirds')) txt+=' · 根据偏好推荐「'+schemes[rec].name+'」，去抽屉切换';
    heroMatchEl.textContent=txt;
  }
  function setBar(nid,bid,val){
    document.getElementById(nid).textContent=val;
    document.getElementById(bid).style.width=val+'%';
  }
  setBar('scPos','scPosBar',sc.pos);
  setBar('scRatio','scRatioBar',sc.ratio);
  setBar('scCons','scConsBar',sc.cons);
  // 冲突
  updateConflictBanner(m);
  // 右侧：图库偏好总览 + 典型照片
  renderComposeSidePanel(profile);
}

function renderComposeSidePanel(profile){
  const schemeNames={thirds:'三分法',golden:'黄金分割',symmetry:'对称',diagonal:'对角线'};
  const overview=computeGalleryOverview(profile);
  if(!overview.length) return;
  const top=overview[0];

  document.getElementById('typicalTitle').textContent = profile ? '典型偏好照' : '典型照片（暂按综合分，评价满 3 次后按偏好推荐）';
  document.getElementById('typicalBody').innerHTML = `
    <img class="typical-img" src="${top.m.dataURL}" alt="${top.m.name}">
    <div class="typical-meta">
      <div class="typical-name">${top.m.name}</div>
      <div class="typical-sub">方案 ${schemeNames[top.scheme]} · 综合分 ${top.total}</div>
      ${top.match!==null
        ? `<div class="typical-match">偏好匹配 ${top.match} / 100</div>`
        : `<div class="typical-match" style="color:var(--ink-3)">偏好学习中…（还需 ${MIN_SAMPLES-PREFS.length} 次评价）</div>`}
    </div>
  `;

  document.getElementById('prefOverviewList').innerHTML = overview.map(({m,scheme,match,total})=>`
    <div class="pref-row${m.id===S.currentId?' active':''}" data-id="${m.id}">
      <img class="pref-row-img" src="${m.dataURL}" alt="${m.name}">
      <div class="pref-row-body">
        <div class="pref-row-name">${m.name}</div>
        <div class="pref-row-sub">${schemeNames[scheme]} · ${m.liked===true?'♥ 喜欢':m.liked===false?'不喜欢':'未评价'}</div>
      </div>
      <span class="pref-row-score">${match!==null?match:total}</span>
    </div>
  `).join('');
  document.querySelectorAll('.pref-row').forEach(row=>{
    row.onclick=()=>{ S.currentId=row.dataset.id; syncComposePage(); };
  });
}

function updateConflictBanner(m){
  const banner=document.getElementById('conflictBanner');
  if(!m||!m.conflicts||!m.conflicts.length){banner.classList.remove('show');return;}
  const cf=m.conflicts[m.conflictIdx||0];
  if(!cf){banner.classList.remove('show');return;}
  banner.classList.add('show');
  document.getElementById('conflictTitle').textContent=
    `色相差异 ${cf.diff}° > 10° — 步骤 C（${(m.conflictIdx||0)+1}/${m.conflicts.length}）`;
  document.getElementById('conflictDesc').textContent=
    `区域 #${cf.ti}：传统「${cf.blockColor}」vs AI「${cf.aiClass}」`;
  document.getElementById('cbTrad').className='cb-btn'+(m.conflictChoice[cf.ti]==='t'?' chosen':'');
  document.getElementById('cbAI').className='cb-btn'+(m.conflictChoice[cf.ti]==='a'?' chosen':'');
}

/* ============================================================
   存图
   ============================================================ */
function exportPng(cvs, filename){
  const a=document.createElement('a');
  a.download=filename||'neatpic.png';
  a.href=cvs.toDataURL('image/png');a.click();
}
function exportJson(data, filename){
  const a=document.createElement('a');
  a.download=filename||'info.json';
  a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  a.click();
}

/* ============================================================
   权重同步（构图台 / 优选台 共用同一份 S.weights）
   ============================================================ */
function applyWeightsFrom(pPct, rPct, cPct){
  const sum=pPct+rPct+cPct||1;
  S.weights={pos:pPct/sum, ratio:rPct/sum, cons:cPct/sum};
  ['wPosD','cwPosD'].forEach(id=>{const el=document.getElementById(id); if(el) el.textContent=S.weights.pos.toFixed(2);});
  ['wRatioD','cwRatioD'].forEach(id=>{const el=document.getElementById(id); if(el) el.textContent=S.weights.ratio.toFixed(2);});
  ['wConsD','cwConsD'].forEach(id=>{const el=document.getElementById(id); if(el) el.textContent=S.weights.cons.toFixed(2);});
  const p=Math.round(S.weights.pos*100), r=Math.round(S.weights.ratio*100), c=Math.round(S.weights.cons*100);
  ['wPos','cwPos'].forEach(id=>{const el=document.getElementById(id); if(el) el.value=p;});
  ['wRatio','cwRatio'].forEach(id=>{const el=document.getElementById(id); if(el) el.value=r;});
  ['wCons','cwCons'].forEach(id=>{const el=document.getElementById(id); if(el) el.value=c;});
}
function syncWeights(){
  applyWeightsFrom(+document.getElementById('wPos').value, +document.getElementById('wRatio').value, +document.getElementById('wCons').value);
  syncComposePage();
}
function syncCurateWeights(){
  applyWeightsFrom(+document.getElementById('cwPos').value, +document.getElementById('cwRatio').value, +document.getElementById('cwCons').value);
  renderCuratePage();
}

/* ============================================================
   识别台（对接 bird_classification_server.py：YOLOv8 检测 + DeepLabV3 分割
   + CLIP 品种识别 + DeepSeek 中文百科/AI 点评 + 懒人模式一键优化）
   ============================================================ */
const BIRD_API='http://localhost:5051';
let recServiceOnline=null;
let lazyModeOn=false;

async function checkRecognizeService(){
  const banner=document.getElementById('recServiceBanner');
  const dot=document.getElementById('recServiceDot');
  const text=document.getElementById('recServiceText');
  if(!banner) return;
  try{
    const r=await fetch(BIRD_API+'/health');
    if(!r.ok) throw new Error('bad status');
    const data=await r.json();
    const ready=data.yolo_loaded && data.deeplab_loaded && data.clip_loaded;
    recServiceOnline=ready;
    if(ready){
      banner.className='service-banner online';
      dot.className='dot on';
      text.textContent='识别服务在线 · 设备 '+data.device;
    }else{
      banner.className='service-banner';
      dot.className='dot busy';
      text.textContent='识别服务已启动，模型加载中…';
    }
  }catch(e){
    recServiceOnline=false;
    banner.className='service-banner offline';
    dot.className='dot';
    text.innerHTML='识别服务未连接 · 请先运行 <code>python3 bird_classification_server.py</code>（默认监听 5051 端口）';
  }
}

async function dataURLToBlob(dataURL){
  const r=await fetch(dataURL);
  return r.blob();
}

async function analyzeOne(m){
  try{
    const blob=await dataURLToBlob(m.dataURL);
    const fd=new FormData();
    fd.append('image', blob, m.name);
    fd.append('lazy_mode', lazyModeOn ? 'true' : 'false');
    const r=await fetch(BIRD_API+'/analyze', {method:'POST', body:fd});
    const data=await r.json();
    if(!r.ok || data.error) throw new Error(data.error||'分析失败');
    m.recognize={
      birds:data.birds, resultImage:data.result_image_data_url,
      critique:data.ai_critique, lazy:data.lazy_mode, ts:Date.now(),
    };
    m._aesthetic=data.aesthetic;
    m.score=deriveScore(m);
  }catch(e){
    m.recognize={error:e.message||String(e)};
  }
  renderRecognizePage();
  // 评分变了，优选台/构图台如果正显示着就顺手刷新
  if(!document.getElementById('page-curate').classList.contains('hidden')) renderCuratePage();
  if(!document.getElementById('page-compose').classList.contains('hidden')) syncComposePage();
}

async function analyzeAllUnrecognized(){
  await checkRecognizeService();
  if(!recServiceOnline) return;
  const targets=GALLERY.filter(m=>!m.recognize || m.recognize.error);
  for(const m of targets){
    m.recognize={pending:true};
    renderRecognizePage();
    await analyzeOne(m);
  }
}

function speciesTally(){
  const tally={};
  GALLERY.forEach(m=>{
    (m.recognize && m.recognize.birds || []).forEach(b=>{
      tally[b.species_en]=(tally[b.species_en]||0)+1;
    });
  });
  return tally;
}

// seenSpecies：跨整页共享的 Set，同一物种的百科介绍只在第一次出现时完整展示，
// 之后再遇到同物种只提示"已在上方展示过"，不重复整段百科文字。
function birdRowHtml(b, seenSpecies){
  const pct=(b.species_confidence*100).toFixed(1);
  const label=b.info && b.info.chinese_name ? `${b.species_en} · ${b.info.chinese_name}` : b.species_en;
  let enc='';
  if(b.info){
    if(seenSpecies && seenSpecies.has(b.species_en)){
      enc=`<div class="rec-encyclopedia rec-encyclopedia-ref">该物种的百科介绍已在上方展示过</div>`;
    }else{
      enc=`<div class="rec-encyclopedia">
        <div><b>学名</b>：${b.info.scientific_name||'—'}</div>
        <div><b>简介</b>：${b.info.introduction||'—'}</div>
        <div><b>栖息地</b>：${b.info.habitat||'—'}</div>
        <div><b>食性</b>：${b.info.diet||'—'}</div>
        <div><b>保护级别</b>：${b.info.conservation_status||'—'}</div>
      </div>`;
      if(seenSpecies) seenSpecies.add(b.species_en);
    }
  }
  return `<div class="rec-bird-row">
    <img class="rec-crop" src="${b.crop_data_url}" alt="${b.species_en}">
    <div class="rec-bird-info">
      <div class="rtk-row">
        <div class="rtk-top"><span class="rtk-label">${label}</span><span class="rtk-val">${pct}%</span></div>
        <div class="rtk-track"><div class="rtk-fill" style="width:${pct}%"></div></div>
      </div>
      ${enc}
    </div>
  </div>`;
}

// DeepSeek 按 prompt 要求把点评包在 ```markdown ... ``` 代码块里，去掉首尾围栏，
// 只留纯 markdown 正文。
function stripMdFence(text){
  const raw=String(text||'').trim();
  if(!raw) return '';
  const fenced=raw.match(/^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$/i);
  if(fenced) return fenced[1].trim();
  if(/^```/.test(raw)) return raw.replace(/^```[a-zA-Z]*\s*/,'').replace(/\s*```$/,'').trim();
  return raw;
}

function escapeHtml(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatInlineMd(text){
  let t=escapeHtml(text);
  t=t.replace(/`([^`]+)`/g,'<code class="md-code">$1</code>');
  t=t.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  t=t.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g,'<em>$1</em>');
  t=t.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a class="md-link" href="$2" target="_blank" rel="noopener">$1</a>');
  return t.replace(/\n/g,'<br>');
}

// 轻量 Markdown 渲染：覆盖标题、加粗、斜体、列表、引用、代码块和分隔线，
// 不引入外部 markdown 库（单文件零依赖）。
function renderMarkdownLite(md){
  const raw=stripMdFence(md);
  const lines=raw.split(/\r?\n/);
  let html='';
  let inList=false; let listType=null; let inCode=false; let codeLines=[];
  const closeList=()=>{ if(inList){ html += listType==='ol' ? '</ol>' : '</ul>'; inList=false; listType=null; } };
  const closeCode=()=>{ if(inCode){ html+=`<pre class="md-pre"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`; inCode=false; codeLines=[]; } };
  for(let i=0;i<lines.length;i++){
    const rawLine=lines[i];
    const line=rawLine.trim();
    if(!line){ closeList(); continue; }
    if(/^```/.test(line)){
      closeList();
      if(inCode) closeCode();
      else { inCode=true; codeLines=[]; }
      continue;
    }
    if(inCode){ codeLines.push(rawLine); continue; }
    const h=/^(#{1,6})\s+(.*)$/.exec(line);
    if(h){ closeList(); const level=Math.min(h[1].length,4); html+=`<h${level} class="md-h">${formatInlineMd(h[2])}</h${level}>`; continue; }
    if(/^>\s+/.test(line)){ closeList(); html+=`<blockquote class="md-blockquote">${formatInlineMd(line.replace(/^>\s+/,''))}</blockquote>`; continue; }
    if(/^[-*]\s+/.test(line)){ if(!inList||listType!=='ul'){ closeList(); html+='<ul class="md-ul">'; inList=true; listType='ul'; } html+=`<li>${formatInlineMd(line.replace(/^[-*]\s+/,''))}</li>`; continue; }
    if(/^\d+[.)]\s+/.test(line)){ if(!inList||listType!=='ol'){ closeList(); html+='<ol class="md-ol">'; inList=true; listType='ol'; } html+=`<li>${formatInlineMd(line.replace(/^\d+[.)]\s+/,''))}</li>`; continue; }
    if(/^[-*_]{3,}$/.test(line)){ closeList(); html+='<hr class="md-hr">'; continue; }
    closeList();
    html+=`<p class="md-p">${formatInlineMd(line)}</p>`;
  }
  closeCode(); closeList();
  return html || '<p class="md-p" style="color:var(--ink-3);">暂无内容</p>';
}

function aestheticBarsHtml(a){
  return `
    <div class="sbr">
      <div class="sbr-top"><span class="sbr-name">构图</span><span class="sbr-val">${a.composition}</span></div>
      <div class="sbr-track"><div class="sbr-fill" style="width:${a.composition}%;background:var(--c-thirds)"></div></div>
    </div>
    <div class="sbr">
      <div class="sbr-top"><span class="sbr-name">色彩丰富度</span><span class="sbr-val">${a.richness}</span></div>
      <div class="sbr-track"><div class="sbr-fill" style="width:${a.richness}%;background:var(--c-golden)"></div></div>
    </div>
    <div class="sbr">
      <div class="sbr-top"><span class="sbr-name">色彩纯净度</span><span class="sbr-val">${a.purity}</span></div>
      <div class="sbr-track"><div class="sbr-fill" style="width:${a.purity}%;background:var(--c-sym)"></div></div>
    </div>`;
}

const REPORT_DRAWER={currentId:null};

function openReportDrawer(m){
  if(!m) return;
  REPORT_DRAWER.currentId=m.id;
  document.getElementById('reportDrawer').classList.add('open');
  renderReportDrawer();
}

function closeReportDrawer(){
  document.getElementById('reportDrawer').classList.remove('open');
  REPORT_DRAWER.currentId=null;
}

function renderReportDrawer(){
  const m=GALLERY.find(x=>x.id===REPORT_DRAWER.currentId);
  if(!m){ return; }
  const mediaSrc=(m.recognize && m.recognize.resultImage) || m.dataURL;
  const birds=(m.recognize && m.recognize.birds) || [];
  const critique=(m.recognize && m.recognize.critique) || '';
  const aesthetic=m._aesthetic;
  const score=m.score;
  const scoreItems=[];
  if(aesthetic){
    scoreItems.push(['构图', aesthetic.composition], ['色彩丰富度', aesthetic.richness], ['色彩纯净度', aesthetic.purity], ['综合', aesthetic.total]);
  }else if(score){
    scoreItems.push(['综合', score.total], ['锚点', score.pos], ['比例', score.ratio], ['一致性', score.cons]);
  }
  document.getElementById('reportDrawerTitle').textContent='照片详情 · '+m.name;
  document.getElementById('reportDrawerBody').innerHTML=`
    <div class="report-hero">
      <div class="report-media"><img src="${mediaSrc}" alt="${m.name}"></div>
      <div class="report-meta">
        <div class="report-title">${m.name}</div>
        <div class="report-sub">查看这张照片的识别结果、评分、DeepSeek 说明，并可在下方调整颜色后导出。</div>
        <div class="report-pill">${aesthetic ? '综合分 '+aesthetic.total : (score ? '综合分 '+score.total : '暂无评分')}</div>
      </div>
    </div>
    <div class="report-section">
      <div class="dl-title">颜色调整</div>
      <div class="edit-preview-wrap">
        <img id="editPreviewImg" class="edit-preview-img" src="${m.dataURL}" alt="${m.name}">
      </div>
      <div class="sl-row">
        <div class="sl-lab"><span>亮度</span><b id="editBrightnessD">${m.edit.brightness}%</b></div>
        <input type="range" id="editBrightness" min="50" max="150" value="${m.edit.brightness}">
      </div>
      <div class="sl-row">
        <div class="sl-lab"><span>对比度</span><b id="editContrastD">${m.edit.contrast}%</b></div>
        <input type="range" id="editContrast" min="50" max="150" value="${m.edit.contrast}">
      </div>
      <div class="sl-row">
        <div class="sl-lab"><span>饱和度</span><b id="editSaturateD">${m.edit.saturate}%</b></div>
        <input type="range" id="editSaturate" min="0" max="200" value="${m.edit.saturate}">
      </div>
      <div class="sl-row">
        <div class="sl-lab"><span>色相偏移</span><b id="editHueD">${m.edit.hue}°</b></div>
        <input type="range" id="editHue" min="-180" max="180" value="${m.edit.hue}">
      </div>
      <div class="btn-row">
        <button class="btn sm" id="editReset">重置</button>
        <button class="btn sm pri" id="editExport"><svg class="ic"><use href="#ic-save"/></svg>导出调色图片</button>
      </div>
    </div>
    <div class="report-section">
      <div class="dl-title">美学得分</div>
      ${scoreItems.length ? `<div class="report-score-grid">${scoreItems.map(([k,v])=>`<div class="report-score-item"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('')}</div>` : '<div class="report-empty">暂无美学评分</div>'}
    </div>
    <div class="report-section">
      <div class="dl-title">识别结果</div>
      ${birds.length ? `<div class="report-birds">${birds.map(b=>`<div class="report-bird"><img src="${b.crop_data_url}" alt="${b.species_en}"><div><div class="rb-name">${b.species_en}</div><div class="rb-meta">置信度：${(b.species_confidence*100).toFixed(1)}%${b.info&&b.info.chinese_name?` · 中文名：${b.info.chinese_name}`:''}${b.info&&b.info.scientific_name?` · 学名：${b.info.scientific_name}`:''}</div></div></div>`).join('')}</div>` : '<div class="report-empty">未检测到鸟类</div>'}
    </div>
    <div class="report-section">
      <div class="dl-title">DeepSeek 说明</div>
      <div class="report-critique">${renderMarkdownLite(stripMdFence(critique))}</div>
    </div>
  `;
  bindEditControls(m);
}

/* ============================================================
   颜色调整与导出（详情抽屉内，per-photo，一定范围内的亮度/对比度/
   饱和度/色相调整，导出走 canvas ctx.filter，与预览所见一致）
   ============================================================ */
function editFilterCss(e){
  return `brightness(${e.brightness}%) contrast(${e.contrast}%) saturate(${e.saturate}%) hue-rotate(${e.hue}deg)`;
}

function bindEditControls(m){
  const img=document.getElementById('editPreviewImg');
  if(!img) return;
  const bEl=document.getElementById('editBrightness'), bD=document.getElementById('editBrightnessD');
  const cEl=document.getElementById('editContrast'), cD=document.getElementById('editContrastD');
  const sEl=document.getElementById('editSaturate'), sD=document.getElementById('editSaturateD');
  const hEl=document.getElementById('editHue'), hD=document.getElementById('editHueD');
  function refresh(){
    bD.textContent=m.edit.brightness+'%'; cD.textContent=m.edit.contrast+'%';
    sD.textContent=m.edit.saturate+'%'; hD.textContent=m.edit.hue+'°';
    img.style.filter=editFilterCss(m.edit);
  }
  bEl.oninput=()=>{ m.edit.brightness=+bEl.value; refresh(); };
  cEl.oninput=()=>{ m.edit.contrast=+cEl.value; refresh(); };
  sEl.oninput=()=>{ m.edit.saturate=+sEl.value; refresh(); };
  hEl.oninput=()=>{ m.edit.hue=+hEl.value; refresh(); };
  document.getElementById('editReset').onclick=()=>{
    m.edit={brightness:100,contrast:100,saturate:100,hue:0};
    bEl.value=100; cEl.value=100; sEl.value=100; hEl.value=0;
    refresh();
  };
  document.getElementById('editExport').onclick=()=>exportEditedImage(m);
  refresh();
}

function exportEditedImage(m){
  const cvs=document.createElement('canvas');
  cvs.width=m.el.naturalWidth; cvs.height=m.el.naturalHeight;
  const ctx=cvs.getContext('2d');
  ctx.filter=editFilterCss(m.edit);
  ctx.drawImage(m.el,0,0,cvs.width,cvs.height);
  exportPng(cvs,'neatpic_edited_'+m.name);
}

function recognizedBodyHtml(m, seenSpecies){
  const birds=m.recognize.birds;
  const a=m._aesthetic;
  let html = birds.length
    ? '<div class="rec-birds">'+birds.map(b=>birdRowHtml(b, seenSpecies)).join('')+'</div>'
    : '<div class="rec-pending">未检测到鸟类</div>';
  if(a){
    html += `<div class="rec-aesthetic">${aestheticBarsHtml(a)}
      <div class="rec-aesthetic-total"><span>综合美学得分</span><b>${a.total}</b></div>
    </div>`;
  }
  if(m.recognize.critique){
    html += `<div class="rec-critique">${renderMarkdownLite(stripMdFence(m.recognize.critique))}</div>`;
  }
  if(a || m.recognize.critique){
    html += `<button class="btn sm" data-report="${m.id}" style="width:100%;margin-bottom:13px;">
      <svg class="ic"><use href="#ic-target"/></svg>查看详情 / 调色导出</button>`;
  }
  if(m.recognize.lazy && m.recognize.lazy.applied){
    html += `<div class="rec-lazy">
      <div class="dl-title">懒人模式：自动裁剪 + 色彩微调</div>
      <img class="rec-lazy-img" src="${m.recognize.lazy.optimized_image_data_url}" alt="优化后">
      <a class="btn sm" href="${m.recognize.lazy.optimized_image_data_url}" download="neatpic_optimized_${m.name}"><svg class="ic"><use href="#ic-save"/></svg>下载优化图</a>
    </div>`;
  }
  return html;
}

// 一张照片可能检测到多只鸟；取置信度最高的一只作为这张照片的"主体物种"，
// 分组、排序都按这个主体物种走。
function primarySpeciesOf(m){
  const birds=(m.recognize && m.recognize.birds) || [];
  if(!birds.length) return null;
  return birds.reduce((best,b)=>(!best||b.species_confidence>best.species_confidence)?b:best, null).species_en;
}

// 把图库按识别状态/物种分桶：已出结果的按主体物种分组（同物种放一起），
// 其余按"待识别/分析中/识别失败/未检测到鸟类"单独归类，不参与物种分组。
function groupGalleryBySpecies(){
  const groups=new Map();
  const notStarted=[], pending=[], errored=[], noBird=[];
  GALLERY.forEach(m=>{
    if(m.recognize && m.recognize.pending){ pending.push(m); return; }
    if(m.recognize && m.recognize.error){ errored.push(m); return; }
    if(m.recognize && m.recognize.birds){
      if(!m.recognize.birds.length){ noBird.push(m); return; }
      const sp=primarySpeciesOf(m);
      if(!groups.has(sp)) groups.set(sp,{species:sp,photos:[]});
      groups.get(sp).photos.push(m);
      return;
    }
    notStarted.push(m);
  });
  const groupList=[...groups.values()].sort((a,b)=>b.photos.length-a.photos.length);
  return {groupList, notStarted, pending, errored, noBird};
}

function renderRecognizePage(){
  const noImg=document.getElementById('recNoImg');
  const hasImg=document.getElementById('recHasImg');
  if(!GALLERY.length){ noImg.style.display=''; hasImg.classList.add('hidden'); return; }
  noImg.style.display='none'; hasImg.classList.remove('hidden');

  const tallyEntries=Object.entries(speciesTally());
  document.getElementById('recTally').textContent = tallyEntries.length
    ? '已识别 '+tallyEntries.length+' 个物种（按检测到的每只鸟统计）：'+tallyEntries.map(([k,v])=>k+'×'+v).join('，')
    : '尚未识别任何照片';

  // 同一物种的百科介绍只展示一次：这个 Set 在整页范围内共享，谁先渲染到谁展示完整版
  const seenSpecies=new Set();

  function photoCardHtml(m){
    let body;
    if(m.recognize && m.recognize.pending){
      body='<div class="rec-pending">分析中…（检测+分割+品种识别+百科+点评，可能需要几秒到十几秒）</div>';
    }else if(m.recognize && m.recognize.error){
      body=`<div class="rec-error">分析失败：${m.recognize.error}</div>
        <button class="btn sm" data-retry="${m.id}" style="margin-top:8px;width:100%;">重试</button>`;
    }else if(m.recognize && m.recognize.birds){
      body=recognizedBodyHtml(m, seenSpecies);
    }else{
      body=`<button class="btn sm" data-recognize="${m.id}" style="width:100%;"><svg class="ic"><use href="#ic-recognize"/></svg>分析此图</button>`;
    }
    const mediaSrc=(m.recognize && m.recognize.resultImage) || m.dataURL;
    return `<div class="rec-card">
      <div class="rec-card-media"><img class="rec-media-img" src="${mediaSrc}" alt="${m.name}"></div>
      <div class="rec-body"><div class="rec-name" title="${m.name}">${m.name}</div>${body}</div></div>`;
  }

  function groupSectionHtml(title, photos){
    if(!photos.length) return '';
    return `<div class="rec-species-group">
      <div class="rec-species-head"><span class="rec-species-name">${title}</span><span class="rec-species-count">${photos.length} 张</span></div>
      <div class="rec-grid">${photos.map(photoCardHtml).join('')}</div>
    </div>`;
  }

  const {groupList, notStarted, pending, errored, noBird} = groupGalleryBySpecies();
  let html='';
  groupList.forEach(g=>{
    const firstBird=g.photos[0].recognize.birds.find(b=>b.species_en===g.species);
    const cname=firstBird && firstBird.info && firstBird.info.chinese_name;
    html += groupSectionHtml(cname?`${g.species} · ${cname}`:g.species, g.photos);
  });
  html += groupSectionHtml('待识别', notStarted);
  html += groupSectionHtml('分析中', pending);
  html += groupSectionHtml('未检测到鸟类', noBird);
  html += groupSectionHtml('识别失败', errored);
  document.getElementById('recGrid').innerHTML = html;

  document.getElementById('recGrid').querySelectorAll('[data-recognize]').forEach(btn=>{
    btn.onclick=()=>{ const m=GALLERY.find(x=>x.id===btn.dataset.recognize); if(m){ m.recognize={pending:true}; renderRecognizePage(); analyzeOne(m); } };
  });
  document.getElementById('recGrid').querySelectorAll('[data-retry]').forEach(btn=>{
    btn.onclick=()=>{ const m=GALLERY.find(x=>x.id===btn.dataset.retry); if(m){ m.recognize={pending:true}; renderRecognizePage(); analyzeOne(m); } };
  });
  document.getElementById('recGrid').querySelectorAll('[data-report]').forEach(btn=>{
    btn.onclick=()=>{ const m=GALLERY.find(x=>x.id===btn.dataset.report); if(m){ openReportDrawer(m); } };
  });
}

/* ============================================================
   优选台（GouTu3 美学评分总览，复用 computeScore，不改动算法本身）
   ============================================================ */
let curateSort='total_desc';
let curateThreshold=60;
const curateSelected=new Set();

function renderCuratePage(){
  const noImg=document.getElementById('curateNoImg');
  const hasImg=document.getElementById('curateHasImg');
  if(!GALLERY.length){ noImg.style.display=''; hasImg.classList.add('hidden'); return; }
  noImg.style.display='none'; hasImg.classList.remove('hidden');

  // 只对已经缓存了后端美学分（m._aesthetic）的照片重算加权总分——纯算术，
  // 不再像以前那样每次渲染/每次拖阈值滑杆都对全部照片重新做像素级 HSV 提取。
  GALLERY.forEach(m=>{ if(m._aesthetic) m.score=deriveScore(m); });

  let list=[...GALLERY];
  if(curateSort==='total_desc') list.sort((a,b)=>(b.score?b.score.total:-1)-(a.score?a.score.total:-1));
  else if(curateSort==='total_asc') list.sort((a,b)=>(a.score?a.score.total:101)-(b.score?b.score.total:101));
  else if(curateSort==='name') list.sort((a,b)=>a.name.localeCompare(b.name));

  const scored=list.filter(m=>m.score);
  const passCount=scored.filter(m=>m.score.total>=curateThreshold).length;
  document.getElementById('curateCount').textContent=`达标 ${passCount} / 共 ${list.length}`;

  document.getElementById('curateGrid').innerHTML = list.map(m=>{
    if(!m.score){
      const checked=curateSelected.has(m.id);
      return `<div class="curate-card below-thresh" data-id="${m.id}">
        <input type="checkbox" class="cc-check" data-check="${m.id}" ${checked?'checked':''}>
        <img class="cc-img" src="${m.dataURL}" alt="${m.name}">
        <span class="cc-score">—</span>
        <div class="cc-body">
          <div class="cc-name" title="${m.name}">${m.name}</div>
          <div style="font-size:11px;color:var(--ink-3);margin-top:5px;">评分中…去识别台分析</div>
        </div>
      </div>`;
    }
    const below = m.score.total < curateThreshold;
    const checked = curateSelected.has(m.id);
    return `<div class="curate-card${below?' below-thresh':''}" data-id="${m.id}">
      <input type="checkbox" class="cc-check" data-check="${m.id}" ${checked?'checked':''}>
      <img class="cc-img" src="${m.dataURL}" alt="${m.name}">
      <span class="cc-score">${m.score.total}</span>
      <div class="cc-body">
        <div class="cc-name" title="${m.name}">${m.name}</div>
        <div class="cc-bars">
          <span style="flex-grow:${m.score.pos||0.01};background:var(--c-thirds)"></span>
          <span style="flex-grow:${m.score.ratio||0.01};background:var(--c-golden)"></span>
          <span style="flex-grow:${m.score.cons||0.01};background:var(--c-sym)"></span>
        </div>
      </div>
    </div>`;
  }).join('');

  document.getElementById('curateGrid').querySelectorAll('[data-check]').forEach(cb=>{
    cb.onclick=e=>e.stopPropagation();
    cb.onchange=()=>{
      if(cb.checked) curateSelected.add(cb.dataset.check); else curateSelected.delete(cb.dataset.check);
    };
  });
  document.getElementById('curateGrid').querySelectorAll('.curate-card').forEach(card=>{
    card.onclick=()=>{ S.currentId=card.dataset.id; goPage('compose'); };
  });
}

/* ============================================================
   导航
   ============================================================ */
const PAGES=['compose','gallery','capture','recognize','curate'];
const PAGE_LABELS={compose:'构图台',gallery:'图库',capture:'采集台',recognize:'识别台',curate:'优选台'};
function goPage(p){
  PAGES.forEach(pg=>document.getElementById('page-'+pg).classList.toggle('hidden',pg!==p));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.toggle('active',b.dataset.page===p));
  const topLabel=document.getElementById('topStageLabel');
  if(topLabel) topLabel.textContent=PAGE_LABELS[p]||p;
  if(p==='compose') syncComposePage();
  if(p==='gallery') renderGallery(document.getElementById('galSearch').value);
  if(p==='recognize'){ renderRecognizePage(); checkRecognizeService(); }
  if(p==='curate') renderCuratePage();
}

/* ============================================================
   上传处理
   ============================================================ */
async function handleFiles(files){
  await addImages(files);
  renderGallery();
  // 如在构图台/识别台/优选台，自动同步
  if(!document.getElementById('page-compose').classList.contains('hidden')) syncComposePage();
  if(!document.getElementById('page-recognize').classList.contains('hidden')) renderRecognizePage();
  if(!document.getElementById('page-curate').classList.contains('hidden')) renderCuratePage();
  // 后台自动开始分析新上传的照片（拿美学评分+品种识别），不阻塞上传流程；
  // 识别服务没启动的话 analyzeAllUnrecognized 会自己检测到并直接跳过。
  analyzeAllUnrecognized();
}

/* ============================================================
   绑定所有事件
   ============================================================ */
function bindAll(){
  // 导航
  document.querySelectorAll('.nav-btn').forEach(btn=>{
    btn.onclick=()=>goPage(btn.dataset.page);
  });

  // 上传
  document.getElementById('btnUploadFiles').onclick=()=>document.getElementById('fileInput').click();
  document.getElementById('btnUploadFolder').onclick=()=>document.getElementById('folderInput').click();
  document.getElementById('fileInput').onchange=e=>{ handleFiles(e.target.files); e.target.value=''; };
  document.getElementById('folderInput').onchange=e=>{ handleFiles(e.target.files); e.target.value=''; };

  // 拖拽
  const dz=document.getElementById('dropZone');
  dz.ondragover=e=>{e.preventDefault();dz.classList.add('drag-over');};
  dz.ondragleave=()=>dz.classList.remove('drag-over');
  dz.ondrop=e=>{e.preventDefault();dz.classList.remove('drag-over');handleFiles(e.dataTransfer.files);};
  dz.onclick=()=>document.getElementById('fileInput').click();

  // 搜索
  document.getElementById('galSearch').oninput=e=>renderGallery(e.target.value);

  // 清空
  document.getElementById('galClearAll').onclick=()=>{
    if(confirm('确认清空图库？')) { GALLERY.length=0; S.currentId=null; renderGallery(); syncComposePage(); }
  };

  // 清除偏好学习
  document.getElementById('galClearPrefs').onclick=()=>{
    if(!confirm('确认清除已学习的偏好数据？')) return;
    PREFS.length=0; savePrefs();
    GALLERY.forEach(m=>{ m.liked=null; });
    renderGallery(document.getElementById('galSearch').value);
    syncComposePage();
    if(document.getElementById('composeDrawer').classList.contains('open')){
      const m=GALLERY.find(x=>x.id===DS.imgId);
      if(m){ updateRateButtons(m); buildDrawerSchemes(); drawerRender(); }
    }
  };

  // 构图台快捷按钮
  document.getElementById('cGoUpload').onclick=()=>document.getElementById('fileInput').click();
  document.getElementById('cGoGallery').onclick=()=>goPage('gallery');
  document.getElementById('tipGoGallery').onclick=()=>goPage('gallery');

  // 识别台
  document.getElementById('btnRecognizeAll').onclick=analyzeAllUnrecognized;
  document.getElementById('recLazyMode').onchange=e=>{ lazyModeOn=e.target.checked; };
  document.getElementById('btnExportRecognize').onclick=()=>{
    const records=GALLERY.map(m=>({
      name:m.name,
      birds:(m.recognize&&m.recognize.birds||[]).map(b=>({
        species_en:b.species_en, chinese_name:b.info&&b.info.chinese_name, confidence:b.species_confidence,
      })),
    }));
    exportJson({records}, 'neatpic_recognize.json');
  };

  // 优选台
  ['cwPos','cwRatio','cwCons'].forEach(id=>{ document.getElementById(id).oninput=syncCurateWeights; });
  document.getElementById('curateSort').onchange=e=>{ curateSort=e.target.value; renderCuratePage(); };
  document.getElementById('curateThreshold').oninput=e=>{
    curateThreshold=+e.target.value;
    document.getElementById('curateThresholdD').textContent=curateThreshold;
    renderCuratePage();
  };
  document.getElementById('curateSelectAll').onclick=()=>{
    GALLERY.forEach(m=>{ if(m.score && m.score.total>=curateThreshold) curateSelected.add(m.id); });
    renderCuratePage();
  };
  document.getElementById('curateExport').onclick=()=>{
    const items=GALLERY.filter(m=>curateSelected.has(m.id) && m.score).map(m=>({
      name:m.name, total:m.score.total, pos:m.score.pos, ratio:m.score.ratio, cons:m.score.cons,
      scheme:m.scheme||'thirds', subjectRatio:m.ratio||20,
    }));
    exportJson({threshold:curateThreshold, weights:S.weights, selected:items}, 'neatpic_curate_selection.json');
  };

  // 冲突按钮
  function nextConflict(choice){
    const m=S.currentId?GALLERY.find(x=>x.id===S.currentId):null;
    if(!m) return;
    const cf=m.conflicts[m.conflictIdx||0];
    if(cf && choice){
      m.conflictChoice[cf.ti]=choice;
      if(choice==='a') m.conflictChoice[cf.ti+'_cls']=cf.aiClass;
    }
    m.conflictIdx=(m.conflictIdx||0)+1;
    syncComposePage();
  }
  document.getElementById('cbTrad').onclick=()=>nextConflict('t');
  document.getElementById('cbAI').onclick=()=>nextConflict('a');
  document.getElementById('cbSkip').onclick=()=>nextConflict(null);

  // 权重
  ['wPos','wRatio','wCons'].forEach(id=>{ document.getElementById(id).oninput=syncWeights; });

  // 存图（构图台）
  document.getElementById('btnSaveImg').onclick=()=>{
    const m=S.currentId?GALLERY.find(x=>x.id===S.currentId):GALLERY[0];
    if(!m) return;
    // 在 offscreen canvas 上重新合成一帧
    const cv2=document.createElement('canvas'); cv2.width=900; cv2.height=600;
    const c2=cv2.getContext('2d');
    const schemes=buildSchemes(); const sc=schemes[m.scheme||'thirds'];
    drawCoverCtx(c2,m.el,900,600);
    const [px,py]=sc.pos;
    const tA=(m.ratio||20)/100*900*600;
    const ar=m.el.naturalWidth/m.el.naturalHeight;
    const bh=Math.sqrt(tA/ar),bw=bh*ar;
    roundClipCtx(c2,m.el,px*900-bw/2,py*600-bh/2,bw,bh,10,900,600);
    drawLinesCtx(c2,sc,m.scheme||'thirds',900,600);
    exportPng(cv2,'neatpic_'+m.name);
  };
  document.getElementById('btnSaveJson').onclick=()=>{
    const m=S.currentId?GALLERY.find(x=>x.id===S.currentId):GALLERY[0];
    if(!m) return;
    const schemes=buildSchemes();const sc=schemes[m.scheme||'thirds'];
    exportJson({composition_method:sc.name,position:sc.pos,
      scheme_score:+sc.score.toFixed(4),aesthetic_score:m.score,
      weights:S.weights,image:m.name},'neatpic_info.json');
  };

  // 报告抽屉
  document.getElementById('reportDrawerClose').onclick=closeReportDrawer;

  // 抽屉
  document.getElementById('drawerClose').onclick=closeDrawer;
  document.getElementById('dRatio').oninput=e=>{
    DS.ratio=+e.target.value;
    document.getElementById('dRatioD').textContent='1/'+(100/DS.ratio).toFixed(1);
    drawerRender();
  };
  // 喜欢/不喜欢 评价
  function rateCurrentDrawerPhoto(liked){
    const m=GALLERY.find(x=>x.id===DS.imgId);
    if(!m) return;
    const next=(m.liked===liked) ? null : liked;
    m.liked=next;
    if(next!==null) pushPrefSample(m, next);
    updateRateButtons(m);
    buildDrawerSchemes();
    drawerRender();
    renderGallery(document.getElementById('galSearch').value);
  }
  document.getElementById('rateLike').onclick=()=>rateCurrentDrawerPhoto(true);
  document.getElementById('rateDislike').onclick=()=>rateCurrentDrawerPhoto(false);
  // 图层
  Object.entries({dlyLines:'lines',dlyAnchor:'anchor',dlyBlocks:'blocks',dlyGrid:'grid'}).forEach(([id,k])=>{
    document.getElementById(id).onchange=e=>{DS.layers[k]=e.target.checked;drawerRender();};
  });

  // 抽屉存图
  document.getElementById('drawerSave').onclick=()=>exportPng(DCV,'neatpic_drawer.png');
  document.getElementById('drawerSaveJson').onclick=()=>{
    const m=GALLERY.find(x=>x.id===DS.imgId);
    const schemes=buildSchemes();const sc=schemes[DS.scheme];
    exportJson({composition_method:sc.name,position:sc.pos,
      scheme_score:+sc.score.toFixed(4),aesthetic_score:m&&m.score,
      target_ratio:'1/'+(100/DS.ratio).toFixed(1),image:m&&m.name},'neatpic_drawer_info.json');
  };
  // 同步到构图台
  document.getElementById('drawerToCompose').onclick=()=>{
    const m=GALLERY.find(x=>x.id===DS.imgId);
    if(m){m.scheme=DS.scheme;m.ratio=DS.ratio;S.currentId=m.id;}
    closeDrawer();
    goPage('compose');
  };
}

/* ============================================================
   启动
   ============================================================ */
(function init(){
  bindAll();
  goPage('compose');
  syncWeights();
})();
