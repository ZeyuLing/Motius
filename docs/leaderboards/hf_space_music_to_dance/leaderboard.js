const fmt=(value,digits=4)=>value==null||!Number.isFinite(Number(value))?"—":Number(value).toFixed(digits);
const metricSpecs=[
  {key:"fid_k",lower:true},{key:"fid_g",lower:true},{key:"diversity_k",lower:false},{key:"diversity_g",lower:false},
  {key:"fid_utmr",lower:true},{key:"beat_align_30fps",lower:false},{key:"jitter",lower:true},{key:"float",lower:true},{key:"slide",lower:true}
];

function rankings(rows){
  const generated=rows.filter(row=>!row.reference);
  const result={};
  for(const spec of metricSpecs){
    const ordered=generated.filter(row=>row[spec.key]!=null).sort((a,b)=>spec.lower?a[spec.key]-b[spec.key]:b[spec.key]-a[spec.key]);
    result[spec.key]=new Map(ordered.map((row,index)=>[row.method,index+1]));
  }
  return result;
}

function methodCell(row,rank){
  const links=[];
  if(row.checkpoint)links.push(`<a href="${row.checkpoint}">HF</a>`);
  if(row.model_card)links.push(`<a href="${row.model_card}">Card</a>`);
  if(row.paper)links.push(`<a href="${row.paper}">Paper</a>`);
  return `<td class="method-cell"><span class="rank ${rank===1?"first":""}">${row.reference?"GT":rank||"·"}</span><strong>${row.method}</strong><span>${row.version}${links.length?` · ${links.join(" / ")}`:""}</span></td>`;
}

function metricCell(row,key,digits,ranks){
  const rank=ranks[key].get(row.method);
  const klass=rank===1?"best":rank===2?"second":"";
  return `<td class="${klass}">${fmt(row[key],digits)}</td>`;
}

function renderRows(data){
  const rows=data.rows,ranks=rankings(rows);
  document.getElementById("method-count").textContent=rows.filter(row=>!row.reference).length;
  document.getElementById("rows").innerHTML=rows.map((row,index)=>{
    const primaryRank=row.reference?null:index;
    return `<tr class="${row.reference?"reference":""}">${methodCell(row,primaryRank)}${metricCell(row,"fid_k",2,ranks)}${metricCell(row,"fid_g",2,ranks)}${metricCell(row,"diversity_k",2,ranks)}${metricCell(row,"diversity_g",2,ranks)}${metricCell(row,"fid_utmr",4,ranks)}${metricCell(row,"beat_align_30fps",4,ranks)}${metricCell(row,"jitter",5,ranks)}${metricCell(row,"float",5,ranks)}${metricCell(row,"slide",5,ranks)}</tr>`;
  }).join("");
}

function renderParity(data){
  document.getElementById("parity-rows").innerHTML=data.paper_parity.map(row=>`<tr><td>${row.row}</td><td>${fmt(row.fid_k,2)}</td><td>${fmt(row.fid_g,2)}</td><td>${fmt(row.diversity_k,2)}</td><td>${fmt(row.diversity_g,2)}</td><td>${fmt(row.beat_align,4)}</td></tr>`).join("");
}

function drawBars(canvas,rows,metrics){
  const ratio=Math.min(window.devicePixelRatio||1,2),rect=canvas.getBoundingClientRect();
  canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));
  const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);ctx.clearRect(0,0,rect.width,rect.height);
  const values=rows.flatMap(row=>metrics.map(metric=>row[metric.key]).filter(value=>value!=null));
  if(!values.length){ctx.fillStyle="#64706b";ctx.font="12px system-ui";ctx.fillText("Reference embeddings are being materialized",8,30);return;}
  const max=Math.max(...values)*1.08||1,left=92,right=18,top=14,rowHeight=(rect.height-top-8)/Math.max(1,rows.length*metrics.length);
  let index=0;
  for(const row of rows){for(const metric of metrics){const value=row[metric.key];if(value==null)continue;const y=top+index*rowHeight;ctx.fillStyle="#64706b";ctx.font="11px system-ui";ctx.textBaseline="middle";ctx.fillText(`${row.method} ${metric.label}`,0,y+rowHeight*.48);ctx.fillStyle=metric.color;ctx.fillRect(left,y+rowHeight*.2,(rect.width-left-right)*(value/max),Math.max(4,rowHeight*.56));ctx.fillStyle="#17201d";ctx.font="10px ui-monospace,monospace";ctx.fillText(fmt(value,metric.digits??2),Math.min(rect.width-right-34,left+(rect.width-left-right)*(value/max)+5),y+rowHeight*.48);index++;}}
}

function renderCharts(data){
  const rows=data.rows;
  drawBars(document.getElementById("official-chart"),rows,[{key:"fid_k",label:"K",color:"#315f9d"},{key:"fid_g",label:"G",color:"#d95f45"}]);
  drawBars(document.getElementById("utmr-chart"),rows,[{key:"fid_utmr",label:"uTMR",color:"#087d72",digits:4}]);
  drawBars(document.getElementById("beat-chart"),rows,[{key:"beat_align_30fps",label:"Beat",color:"#a87312",digits:4}]);
}

async function init(){
  const response=await fetch("music_to_dance_results.json",{cache:"no-cache"});if(!response.ok)throw new Error(`Results ${response.status}`);const data=await response.json();
  renderRows(data);renderParity(data);renderCharts(data);
  document.getElementById("updated").textContent=`Updated ${data.updated}`;document.getElementById("footer-date").textContent=data.updated;
  document.getElementById("beat-audit").textContent="Audit: Motius 60 fps exactly matches the released Bailando implementation (maximum per-case delta 2.3e-15). Standardizing the same 40 predictions to 30 fps changes the mean from 0.226810 to 0.227061; the paper gap is therefore not caused by frame-rate handling.";
  window.addEventListener("resize",()=>renderCharts(data));
}
init().catch(error=>{document.getElementById("updated").textContent=error.message;console.error(error)});
