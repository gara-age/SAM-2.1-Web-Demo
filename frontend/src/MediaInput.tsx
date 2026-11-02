import {useRef} from 'react';
import {Upload,RotateCcw} from 'lucide-react';

type Props={disabled:boolean;kind?:string;name?:string|null;index?:number;total?:number;fps?:number;ended?:boolean;onFile:(file:File)=>Promise<void>;onRewind:()=>Promise<void>};
export function MediaInput({disabled,kind,name,index=0,total=0,fps=0,ended,onFile,onRewind}:Props){
  const input=useRef<HTMLInputElement>(null);
  return <div className="media-input">
    <input ref={input} type="file" hidden aria-label="이미지 또는 동영상 파일" accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.avi,.mkv,.m4v,.webm" disabled={disabled} onChange={e=>{const file=e.target.files?.[0];e.target.value='';if(file)void onFile(file);}}/>
    <button className="button full" disabled={disabled} onClick={()=>input.current?.click()}><Upload size={14}/> 이미지·동영상 불러오기</button>
    <small>최대 250MB · 원본 해상도 사용 · 불러오면 현재 선택을 초기화합니다.</small>
    {name&&<div className="media-details"><strong title={name}>{name}</strong>{kind==='video'?<><span>{ended?'영상 끝':'영상 위치'} · {index+1}{total>0?` / ${total}`:''} 프레임 · {fps>0?`${(index/fps).toFixed(2)}초`:''}</span><p>첫 화면에서 대상을 선택하고 추적을 시작하세요. 연속 모드는 원본 속도로 재생하며 추적합니다. SAM 순차 모드는 AI 처리 속도로 진행합니다.</p><button className="button full" disabled={disabled} onClick={()=>void onRewind()}><RotateCcw size={13}/> 처음으로 · 선택 초기화</button></>:<p>정지 이미지 · 클릭 또는 텍스트로 선택하고 좌표와 크기를 확인하세요.</p>}</div>}
  </div>;
}
