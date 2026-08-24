const fs=require('fs');
const js=fs.readFileSync(process.argv[2],'utf8');
const el=()=>({classList:{toggle(){},add(){},remove(){},contains(){return false}},
  style:{}, children:[], appendChild(){}, querySelector:()=>el(), querySelectorAll:()=>[],
  addEventListener(){}, removeAttribute(){}, set innerHTML(v){}, get innerHTML(){return ''},
  set textContent(v){}, get textContent(){return ''}, set src(v){}, set onclick(v){},
  set onerror(v){}, set onload(v){}, get clientWidth(){return 100}, childElementCount:0});
global.document={getElementById:()=>el(), createElement:()=>el(), addEventListener(){}};
global.fetch=async()=>({json:async()=>({branch:'X',branches:['X'],cameras:[],events:[],
  online:0,rules:0,detectors:0,locked:false,lock_left:0,reachable:true,scanning:false,
  scanned_ago:1,local:false,people:[]}), ok:true, status:200,
  headers:{get:()=>'1'}, blob:async()=>({})});
global.setInterval=()=>{}; global.setTimeout=()=>{};
global.URL={createObjectURL:()=>'x',revokeObjectURL(){}};
global.confirm=()=>false;
try { new Function(js)(); console.log('OK'); }
catch(e){ console.log('XATO: '+e.constructor.name+': '+e.message); process.exit(1); }
