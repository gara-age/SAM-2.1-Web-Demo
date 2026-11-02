import {useCallback,useEffect,useRef,useState} from 'react';
import {ArrowDownToLine,Box,Camera,Check,ChevronDown,LoaderCircle,Plus,Square,Trash2} from 'lucide-react';
import {ScanViewer} from './ScanViewer';

type Capture={name:string;frame_id:number;sharpness:number;thumbnail:string};
type Scan={id:string;object_id:number;object_name:string;source:string;created_at:number;state:string;stage:string;error:string|null;captures:Capture[];count:number;minimum_views:number;maximum_views:number;engine_available:boolean;registered_views:number;vertices:number;faces:number;points?:number;model_url:string|null;version:number;progress?:number;units:string};
type Props={selected:{id:number;name:string;area:number}|undefined;frameId:number|null;connected:boolean;onStarted:()=>void};
async function request(path:string,body?:unknown,method='POST'){
  const response=await fetch('/api/scans'+path,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  const data=await response.json();if(response.status===404)throw new Error('실행 중인 서버에 3D 스캔 API가 없습니다. 최신 start.cmd로 서버를 다시 시작하세요.');if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'스캔 요청을 처리할 수 없습니다.');return data;
}

export function ScanPanel({selected,frameId,connected,onStarted}:Props){
  const [sessions,setSessions]=useState<Scan[]>([]),[scanId,setScanId]=useState<string|null>(null),[scan,setScan]=useState<Scan|null>(null);
  const [busy,setBusy]=useState(false),[auto,setAuto]=useState(false),[error,setError]=useState<string|null>(null),[notice,setNotice]=useState<string|null>(null);
  const inFlight=useRef(false),frame=useRef(frameId),idRef=useRef(scanId),lastAttempt=useRef<number|null>(null);
  frame.current=frameId;idRef.current=scanId;
  useEffect(()=>{let alive=true;request('',undefined,'GET').then((data:Scan[])=>{if(!alive)return;setSessions(data);const saved=localStorage.getItem('sam.scan.session');setScanId(data.some(s=>s.id===saved)?saved:data[0]?.id??null);}).catch(e=>{if(alive)setError(e.message);});return()=>{alive=false;};},[]);
  useEffect(()=>{if(!scanId){setScan(null);return;}localStorage.setItem('sam.scan.session',scanId);let alive=true;const poll=()=>request('/'+scanId,undefined,'GET').then((data:Scan)=>{if(alive){setScan(data);setSessions(old=>[data,...old.filter(s=>s.id!==data.id)]);}}).catch(e=>{if(alive)setError(e.message);});void poll();const timer=setInterval(poll,1800);return()=>{alive=false;clearInterval(timer);};},[scanId]);
  const capture=useCallback(async(automatic=false)=>{
    const currentId=idRef.current,currentFrame=frame.current;if(!currentId||currentFrame==null||inFlight.current)return;
    if(automatic&&lastAttempt.current===currentFrame)return;lastAttempt.current=currentFrame;
    inFlight.current=true;if(!automatic)setBusy(true);
    try{const data=await request(`/${currentId}/capture`,{frame_id:currentFrame});if(idRef.current===currentId){setScan(data);setNotice(`프레임 #${currentFrame} 촬영 완료`);setError(null);}}
    catch(e){if(idRef.current===currentId){if(automatic)setNotice((e as Error).message);else setError((e as Error).message);}}
    finally{inFlight.current=false;setBusy(false);}
  },[]);
  useEffect(()=>{if(!auto||!connected||scan?.state==='reconstructing')return;const timer=setInterval(()=>void capture(true),1800);return()=>clearInterval(timer);},[auto,connected,scan?.state,capture]);
  useEffect(()=>{if(scan?.state==='reconstructing'||(scan&&scan.count>=scan.maximum_views))setAuto(false);},[scan?.state,scan?.count]);
  async function action(task:()=>Promise<void>){if(busy||inFlight.current)return;setBusy(true);setError(null);setNotice(null);try{await task();}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  const working=scan?.state==='reconstructing';
  return <section className="scan-panel panel" aria-label="3D 스캔"><div className="panel-heading"><Box size={17}/><h2>3D 오브젝트 스캔</h2><span className="scan-beta">MULTI VIEW</span></div>
    <ScanViewer url={scan?.model_url??null} version={scan?.version??0} label={scan?`${scan.object_name}${scan.state!=='complete'&&scan.model_url?' · 이전 복원 결과':''}`:'3D SCAN'}/>
    <div className="scan-controls"><div className="scan-session-row"><select aria-label="저장된 3D 스캔" value={scanId??''} disabled={busy||!!working||auto} onChange={e=>{setScanId(e.target.value||null);setError(null);setNotice(null);}}><option value="">저장된 스캔 선택</option>{sessions.map(s=><option key={s.id} value={s.id}>{s.object_name} · {s.source==='synthetic-3d-fixture'?'합성 검증 · ':''}{new Date(s.created_at*1000).toLocaleString('ko-KR',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})}</option>)}</select><button className="button" title="선택한 객체로 새 스캔 시작" aria-label="새 3D 스캔" disabled={busy||!!working||!connected||!selected?.area} onClick={()=>action(async()=>{setAuto(false);const data=await request('',{object_id:selected!.id});setScan(data);setScanId(data.id);setSessions(old=>[data,...old]);onStarted();setNotice('촬영을 시작했습니다. 물체 형태를 유지하고 카메라를 조금씩 이동하세요.');})}><Plus size={14}/> 새 스캔</button></div>
    {!connected?<p className="scan-prerequisite">카메라를 시작하고 객체를 선택한 뒤 새 스캔을 만드세요.</p>:!selected?.area?<p className="scan-prerequisite">분할된 객체를 객체 목록에서 선택하면 새 스캔 버튼이 활성화됩니다.</p>:<p className="scan-prerequisite">현재 선택한 {selected.name}을 촬영하려면 새 스캔을 누르세요.</p>}<div className="scan-target"><span>{scan?`촬영 대상 · ${scan.object_name}`:selected?`선택됨 · ${selected.name}`:'먼저 영상에서 객체를 선택하세요'}</span><span>{scan?.count??0}<small> / {scan?.maximum_views??120} 장</small></span></div>
    {scan?.source==='synthetic-3d-fixture'&&<div className="scan-fixture-label">검증용 합성 3D 물체 · 실제 COLMAP 복원 결과</div>}
    <div className="scan-capture-buttons"><button className="button" disabled={busy||!!working||!scan||!connected||frameId==null||scan.count>=scan.maximum_views} onClick={()=>capture()}><Camera size={14}/> 현재 각도 촬영</button><button className={`button ${auto?'auto-on':''}`} aria-pressed={auto} disabled={busy||!!working||!scan||!connected||scan.count>=scan.maximum_views} onClick={()=>{setAuto(!auto);lastAttempt.current=null;setNotice(null);}}><span className={auto?'record-dot recording':'record-dot'}/>{auto?'자동 촬영 중':'자동 촬영'}</button></div>
    <div className="scan-progress"><div style={{width:`${Math.min((scan?.count??0)/30*100,100)}%`}}/></div><div className="scan-count-caption"><span>최소 12장 · 권장 30–60장</span><span>조금씩 이동 · 충분한 겹침</span></div>
    {scan&&scan.captures.length>0&&<div className="scan-thumbnails">{scan.captures.map(c=><div key={c.name}><img src={c.thumbnail} alt={`촬영 프레임 ${c.frame_id}`} loading="lazy"/><span>#{c.frame_id}</span><button title="촬영 목록에서 제외" aria-label={`프레임 ${c.frame_id} 촬영 제외`} disabled={busy||!!working||auto} onClick={()=>action(async()=>setScan(await request(`/${scan.id}/captures/${c.name}`,undefined,'DELETE')))}><Trash2 size={10}/></button></div>)}</div>}
    {notice&&<p className="scan-notice" role="status">{notice}</p>}{(error||scan?.error)&&<p className="scan-error" role="alert">{error??scan?.error}</p>}
    <button className="button primary full scan-build" disabled={busy||!scan||!!working||scan.count<scan.minimum_views||!scan.engine_available} onClick={()=>action(async()=>{setAuto(false);setScan(await request(`/${scan!.id}/reconstruct`));})}>{working?<LoaderCircle size={14} className="spin"/>:<Box size={15}/>} {working?'3D 형상 복원 중…':scan&&scan.count<scan.minimum_views?`3D 복원 · ${scan.minimum_views-scan.count}장 더 촬영`:'촬영 이미지로 3D 복원'}</button>
    {scan&&<div className="scan-job-status"><span>{scan.state==='complete'?<Check size={12}/>:working?<LoaderCircle size={12} className="spin"/>:<span className="local-dot"/>}{scan.stage}</span>{working&&<button className="text-button" disabled={busy} onClick={()=>action(async()=>setScan(await request(`/${scan.id}/cancel`)))}><Square size={10}/> 중지</button>}</div>}
    {working&&<div className="scan-job-progress"><i style={{width:`${scan.progress??0}%`}}/></div>}
    {scan?.model_url&&<><div className="scan-model-stats"><span><b>{scan.registered_views}</b> 연결 사진</span><span><b>{scan.vertices.toLocaleString()}</b> 정점</span><span><b>{scan.faces.toLocaleString()}</b> 면</span></div><div className="scan-downloads"><a href={scan.model_url} className="button" download><ArrowDownToLine size={12}/> PLY 모델</a><a href={scan.model_url.replace('model.ply','points.ply')} className="button" download><ArrowDownToLine size={12}/> 점군</a></div></>}
    {scan&&(working||scan.error||scan.model_url)&&<a className="scan-log" href={`/api/scans/${scan.id}/log.txt`} target="_blank" rel="noreferrer">복원 로그 확인 ↗</a>}
    <details className="scan-help"><summary><ChevronDown size={11}/> 촬영 방법과 스캔 안내</summary><ol><li>물체를 선택하고 + 버튼으로 스캔을 시작합니다.</li><li>물체는 그대로 두고, 카메라를 조금씩 옮겨 앞·옆·위쪽을 촬영하세요. 촬영 사이에 겹치는 부분이 충분해야 합니다.</li><li>자동 촬영은 새 분석 프레임만 저장합니다. 마스크가 어긋나면 멈추고 보정하세요.</li><li>12장 이상 모은 뒤 3D 복원을 누릅니다. 복원 중에는 추적을 일시정지합니다.</li></ol><p>반사·투명·단색 물체는 복원이 어려울 수 있습니다. 촬영하지 못한 면은 누락되거나 표면 생성 과정에서 보간될 수 있습니다. 결과는 상대 크기의 스캔 메시이며 정밀 치수 CAD가 아닙니다.</p><p>촬영과 모델은 D:\SAM2TrackingWeb\data\scans에 저장됩니다.</p></details>
    </div></section>;
}
