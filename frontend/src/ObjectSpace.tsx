import {useState} from 'react';
import {Layers} from 'lucide-react';
import {ObjectPlane} from './ObjectPlane';
import {ScanPanel} from './ScanPanel';
type Props={frame?:{id:number;width:number;height:number;jpeg:string};selected?:{id:number;name:string;area:number;mask_png:string|null;bbox:number[]|null};connected:boolean;onStarted:()=>void};
export function ObjectSpace({frame,selected,connected,onStarted}:Props){
  const [mode,setMode]=useState<'plane'|'scan'>('plane');
  return <section aria-label="선택 객체 공간 보기"><div className="space-tabs" role="group" aria-label="객체 공간 표시 방식"><button aria-pressed={mode==='plane'} onClick={()=>setMode('plane')}>이미지 평면 · 바로 보기</button><button aria-pressed={mode==='scan'} onClick={()=>setMode('scan')}>다중 시점 3D 스캔</button></div>
    {mode==='plane'?<div className="panel"><div className="panel-heading"><Layers size={16}/><h2>선택 객체 · 이미지 평면</h2></div><ObjectPlane frame={frame} object={selected}/><p className="space-note">선택한 프레임의 마스크로 배경을 제거한 평면 이미지입니다. 선택·추적 결과에 맞춰 갱신되며 실제 깊이나 뒷면을 복원하지 않습니다.</p></div>:<ScanPanel selected={selected} frameId={frame?.id??null} connected={connected} onStarted={onStarted}/>}</section>;
}
