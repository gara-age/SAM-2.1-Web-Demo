import {useEffect,useRef,useState} from 'react';
import {Layers,RefreshCw} from 'lucide-react';
import {ObjectPlane,type PreviewFrame,type PreviewObject} from './ObjectPlane';
import {ScanPanel} from './ScanPanel';
import {DepthCloud} from './DepthCloud';
type Props={frame?:PreviewFrame;selected?:PreviewObject;connected:boolean;onStarted:()=>void};
export function ObjectSpace({frame,selected,connected,onStarted}:Props){
  const [mode,setMode]=useState<'depth'|'plane'|'scan'>('depth'),[live,setLive]=useState(false);
  const latest=useRef({frame,selected});latest.current={frame,selected};
  const [snapshot,setSnapshot]=useState({frame,selected});
  useEffect(()=>{if(selected?.id!==snapshot.selected?.id||(!snapshot.selected?.mask_png&&selected?.mask_png))setSnapshot({frame,selected});},[selected?.id,selected?.mask_png,snapshot.selected?.id,snapshot.selected?.mask_png]);
  useEffect(()=>{if(!live)return;const timer=setInterval(()=>setSnapshot(latest.current),250);return()=>clearInterval(timer);},[live]);
  return <section aria-label="선택 객체 공간 보기"><div className="space-tabs" role="group" aria-label="객체 공간 표시 방식"><button aria-pressed={mode==='depth'} onClick={()=>setMode('depth')}>실시간 깊이·점군</button><button aria-pressed={mode==='plane'} onClick={()=>setMode('plane')}>평면 이미지</button><button aria-pressed={mode==='scan'} onClick={()=>setMode('scan')}>3D 스캔</button></div>
    {mode==='depth'?<div className="panel"><div className="panel-heading"><Layers size={16}/><h2>선택 객체 · 실시간 3D 표면</h2></div><DepthCloud objectId={selected?.id} available={connected&&!!selected?.mask_png&&selected.area>0}/></div>:mode==='plane'?<div className="panel"><div className="panel-heading"><Layers size={16}/><h2>선택 객체 · 이미지 평면</h2></div><ObjectPlane frame={snapshot.frame} object={snapshot.selected} mode="plane"/>
      <div className="space-preview-controls"><button className="text-button" disabled={!selected?.mask_png} onClick={()=>setSnapshot(latest.current)}><RefreshCw size={12}/> 현재 모습 갱신</button><label><input type="checkbox" checked={live} onChange={e=>setLive(e.target.checked)}/> 자동 갱신 · 최대 4FPS</label></div>
      <p className="space-note">배경을 제거한 두께 없는 평면 이미지입니다. 기본은 선택 순간의 모습이며, 미리보기는 별도로 갱신합니다.</p></div>:<ScanPanel selected={selected} frameId={frame?.id??null} connected={connected} onStarted={onStarted}/>}</section>;
}
