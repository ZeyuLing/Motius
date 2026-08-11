const fmt=(value,digits=4)=>value==null||!Number.isFinite(Number(value))?"—":Number(value).toFixed(digits);
const metricSpecs=[
  {key:"fgd",lower:true},{key:"bc",lower:false},{key:"diversity",lower:false},
  {key:"utmr_fid",lower:true},{key:"utmr_paired_distance",lower:true},
  {key:"rotation_geodesic",lower:true},{key:"expression_l2",lower:true},{key:"translation_l2",lower:true}
];

function eligible(row){return row.kind==="method"&&row.verified===true&&row.utmr_fid!=null}
function rankingMaps(rows){
  const result={};
  for(const metric of metricSpecs){
    const ranked=rows.filter(row=>eligible(row)&&row[metric.key]!=null).sort((a,b)=>metric.lower?a[metric.key]-b[metric.key]:b[metric.key]-a[metric.key]);
    result[metric.key]=new Map(ranked.map((row,index)=>[row.method,index+1]));
  }
  return result;
}
function links(row){
  const values=[];
  if(row.checkpoint)values.push(`<a href="${row.checkpoint}" target="_blank">checkpoint</a>`);
  if(row.model_card)values.push(`<a href="${row.model_card}" target="_blank">card</a>`);
  if(row.paper)values.push(`<a href="${row.paper}" target="_blank">paper</a>`);
  return values.length?` · ${values.join(" / ")}`:"";
}
function status(row){
  const label=row.kind==="reference"?"reference":row.submission_status.replaceAll("_"," ");
  return `<span class="status ${row.verified?"verified":""}">${label}</span>`;
}
function metricCell(row,key,digits,ranks){
  const rank=ranks[key].get(row.method);
  return `<td class="${rank===1?"best":""}">${fmt(row[key],digits)}</td>`;
}
function renderPaperParity(data){
  const reference=data.paper_reference;
  const measured=data.rows.find(row=>row.method==="Language of Motion"&&row.verified);
  if(!reference||!measured)return;
  const specs=[
    {key:"fgd",digits:4},{key:"bc",digits:4},{key:"diversity",digits:3}
  ];
  const cells=row=>specs.map(spec=>`<td>${fmt(row[spec.key],spec.digits)}</td>`).join("");
  const deltas=Object.fromEntries(specs.map(spec=>[
    spec.key,
    100*(Number(measured[spec.key])-Number(reference[spec.key]))/Number(reference[spec.key])
  ]));
  document.getElementById("parity-rows").innerHTML=`
    <tr><td><strong>Paper</strong><span>${reference.source}</span></td>${cells(reference)}</tr>
    <tr><td><strong>Motius</strong><span>public demo checkpoint · 15 verified clips</span></td>${cells(measured)}</tr>
    <tr class="delta"><td><strong>Relative difference</strong><span>measured minus paper</span></td>${specs.map(spec=>`<td>${deltas[spec.key]>=0?"+":""}${fmt(deltas[spec.key],2)}%</td>`).join("")}</tr>`;
  const maximum=Math.max(...Object.values(deltas).map(Math.abs));
  document.getElementById("parity-note").textContent=
    `All three metrics are within ${maximum.toFixed(2)}% of the paper row, but this remains near parity rather than an exact paper-checkpoint reproduction.`;
}
function render(data){
  const ranks=rankingMaps(data.rows);
  const ranked=data.rows.filter(eligible);
  const displayed=[
    ...data.rows.filter(row=>row.kind==="reference"),
    ...data.rows.filter(row=>row.kind!=="reference").sort((a,b)=>{
      const aRank=ranks.utmr_fid.get(a.method)??Number.MAX_SAFE_INTEGER;
      const bRank=ranks.utmr_fid.get(b.method)??Number.MAX_SAFE_INTEGER;
      return aRank-bRank;
    }),
  ];
  document.getElementById("method-count").textContent=ranked.length;
  document.getElementById("snapshot").textContent=data.updated==="pending-evaluation"?"Evaluation in progress":`Updated ${data.updated}`;
  document.getElementById("updated").textContent=`BEAT2 · ${data.updated}`;
  document.getElementById("rows").innerHTML=displayed.map(row=>{
    const primary=ranks.utmr_fid.get(row.method);
    const marker=row.kind==="reference"?"GT":primary||"·";
    return `<tr class="${row.kind==="reference"?"reference":""}"><td class="method"><span class="rank">${marker}</span><strong>${row.method}</strong><small>${row.version}${links(row)}</small></td><td>${status(row)}</td>${metricCell(row,"fgd",4,ranks)}${metricCell(row,"bc",4,ranks)}${metricCell(row,"diversity",2,ranks)}${metricCell(row,"utmr_fid",4,ranks)}${metricCell(row,"utmr_paired_distance",4,ranks)}${metricCell(row,"rotation_geodesic",5,ranks)}${metricCell(row,"expression_l2",5,ranks)}${metricCell(row,"translation_l2",5,ranks)}</tr>`;
  }).join("");
  renderPaperParity(data);
}
fetch("speech_to_gesture_results.json",{cache:"no-cache"})
  .then(response=>{if(!response.ok)throw new Error(`Results ${response.status}`);return response.json()})
  .then(render)
  .catch(error=>{document.getElementById("snapshot").textContent=error.message;console.error(error)});
