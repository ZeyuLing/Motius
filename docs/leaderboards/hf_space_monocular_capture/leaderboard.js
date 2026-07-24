const protocolSpecs={
  "3dpw_test_camera_v1":{
    label:"3DPW Test · Camera",
    metrics:[
      ["pa_mpjpe_mm","PA-MPJPE ↓",2],
      ["mpjpe_mm","MPJPE ↓",2],
      ["pve_mm","PVE ↓",2],
      ["accel_mps2","Accel ↓",3]
    ]
  },
  "emdb_1_camera_v1":{
    label:"EMDB-1 · Camera",
    metrics:[
      ["pa_mpjpe_mm","PA-MPJPE ↓",2],
      ["mpjpe_mm","MPJPE ↓",2],
      ["pve_mm","PVE ↓",2],
      ["accel_mps2","Accel ↓",3]
    ]
  },
  "emdb_2_global_v1":{
    label:"EMDB-2 · World",
    metrics:[
      ["w_mpjpe_mm","W-MPJPE ↓",2],
      ["wa_mpjpe_mm","WA-MPJPE ↓",2],
      ["rte_percent","RTE ↓",3],
      ["jitter_mps3","Jitter ↓",3],
      ["foot_sliding_mm","FS ↓",2]
    ]
  }
};
const fmt=(value,digits)=>value==null||!Number.isFinite(Number(value))?"—":Number(value).toFixed(digits);
const safe=value=>String(value??"").replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
let results=null;

function eligible(row,protocol){
  return row.verified===true&&row.protocols?.[protocol]?.metrics;
}
function renderMethods(methods){
  document.getElementById("method-inventory").innerHTML=methods.map(method=>`
    <article>
      <div class="method-mark">${safe(method.body_model)}</div>
      <h3>${safe(method.method)}</h3>
      <p>${safe(method.implementation_status.replaceAll("_"," "))}</p>
      <div class="method-links"><a href="${method.source}" target="_blank" rel="noopener noreferrer">source</a><a href="${method.paper}" target="_blank" rel="noopener noreferrer">paper</a>${method.model_card?`<a href="${method.model_card}" target="_blank" rel="noopener noreferrer">card</a>`:""}</div>
    </article>`).join("");
}
function renderLeaderboard(protocol){
  const spec=protocolSpecs[protocol];
  const rows=results.rows.filter(row=>eligible(row,protocol));
  const sorted=[...rows].sort((a,b)=>{
    const key=spec.metrics[0][0];
    return a.protocols[protocol].metrics[key]-b.protocols[protocol].metrics[key];
  });
  document.getElementById("metric-head").innerHTML=`<th>Method</th><th>Status</th><th>Coverage ↑</th>${spec.metrics.map(([,label])=>`<th>${label}</th>`).join("")}`;
  document.getElementById("rows").innerHTML=sorted.length?sorted.map((row,index)=>{
    const run=row.protocols[protocol];
    return `<tr><td class="method"><span class="rank">${index+1}</span><strong>${safe(row.method)}</strong><small>${safe(row.version)} · ${safe(row.body_model)}</small></td><td><span class="status verified">verified</span></td><td>${fmt(run.coverage_percent,2)}%</td>${spec.metrics.map(([key,,digits])=>`<td class="${index===0?"best":""}">${fmt(run.metrics[key],digits)}</td>`).join("")}</tr>`;
  }).join(""):`<tr class="empty"><td colspan="${3+spec.metrics.length}"><strong>No verified rows yet.</strong><span>Licensed 3DPW/EMDB manifests and full Motius runs are required before ranking opens.</span></td></tr>`;
  document.getElementById("method-count").textContent=sorted.length;
  document.getElementById("active-protocol").textContent=spec.label;
}
function render(data){
  results=data;
  const select=document.getElementById("protocol-select");
  select.innerHTML=Object.entries(protocolSpecs).map(([key,value])=>`<option value="${key}">${value.label}</option>`).join("");
  select.addEventListener("change",event=>renderLeaderboard(event.target.value));
  renderMethods(data.methods);
  renderLeaderboard(select.value);
  document.getElementById("snapshot").textContent=data.updated==="pending-licensed-data"?"Awaiting licensed test data":`Updated ${data.updated}`;
  document.getElementById("updated").textContent=data.updated;
}

fetch("monocular_capture_results.json",{cache:"no-cache"})
  .then(response=>{if(!response.ok)throw new Error(`Results ${response.status}`);return response.json()})
  .then(render)
  .catch(error=>{document.getElementById("snapshot").textContent=error.message;console.error(error)});
