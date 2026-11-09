import {useEffect,useRef,useState} from 'react';
import {Layers,RefreshCw} from 'lucide-react';
import {ObjectPlane,type PreviewFrame,type PreviewObject} from './ObjectPlane';
import {ScanPanel} from './ScanPanel';
type Props={frame?:PreviewFrame;selected?:PreviewObject;connected:boolean;onStarted:()=>void};
export function ObjectSpace({frame,selected,connected,onStarted}:Props){
  const [mode,setMode]=useState<'volume'|'plane'|'scan'>('volume'),[depth,setDepth]=useState(.28),[live,setLive]=useState(false);
  const latest=useRef({frame,selected});latest.current={frame,selected};
  const [snapshot,setSnapshot]=useState({frame,selected});
  useEffect(()=>{if(selected?.id!==snapshot.selected?.id||(!snapshot.selected?.mask_png&&selected?.mask_png))setSnapshot({frame,selected});},[selected?.id,selected?.mask_png,snapshot.selected?.id,snapshot.selected?.mask_png]);
  useEffect(()=>{if(!live)return;const timer=setInterval(()=>setSnapshot(latest.current),250);return()=>clearInterval(timer);},[live]);
  return <section aria-label="선택 객체 공간 보기"><div className="space-tabs" role="group" aria-label="객체 공간 표시 방식"><button aria-pressed={mode==='volume'} onClick={()=>setMode('volume')}>입체 미리보기</button><button aria-pressed={mode==='plane'} onClick={()=>setMode('plane')}>평면 이미지</button><button aria-pressed={mode==='scan'} onClick={()=>setMode('scan')}>3D 스캔</button></div>
    {mode!=='scan'?<div className="panel"><div className="panel-heading"><Layers size={16}/><h2>선택 객체 · {mode==='volume'?'추정 입체':'이미지 평면'}</h2></div><ObjectPlane frame={snapshot.frame} object={snapshot.selected} mode={mode} depth={depth}/>
      <div className="space-preview-controls"><button className="text-button" disabled={!selected?.mask_png} onClick={()=>setSnapshot(latest.current)}><RefreshCw size={12}/> 현재 모습 갱신</button><label><input type="checkbox" checked={live} onChange={e=>setLive(e.target.checked)}/> 자동 갱신 · 최대 4FPS</label>{mode==='volume'&&<label>두께 <input aria-label="추정 입체 두께" type="range" min="0.06" max="0.6" step="0.02" value={depth} onChange={e=>setDepth(Number(e.target.value))}/></label>}</div>
      <p className="space-note">{mode==='volume'?'마스크 윤곽으로 앞·뒤 곡면과 옆면을 만든 근사 모델입니다. 앞면은 원본 사진, 보이지 않는 뒤·옆면은 중립색으로 표시합니다. 실제 스캔 형상은 아닙니다.':'배경을 제거한 두께 없는 평면 이미지입니다.'} 기본은 선택 순간의 모습이며, 추적 영상 속도를 위해 미리보기는 별도로 갱신합니다.</p></div>:<ScanPanel selected={selected} frameId={frame?.id??null} connected={connected} onStarted={onStarted}/>}</section>;
}
