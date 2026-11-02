import {useState} from 'react';
import {MessageSquare,Search} from 'lucide-react';

type Props={model:string;disabled:boolean;hasFrame:boolean;onSelect:(text:string)=>Promise<void>};
export function TextSelection({model,disabled,hasFrame,onSelect}:Props){
  const [text,setText]=useState('사람 선택해');
  if(model!=='sam3.1')return null;
  return <div className="text-selection"><label><MessageSquare size={12}/> 텍스트로 객체 선택</label><form onSubmit={e=>{e.preventDefault();void onSelect(text);}}><input aria-label="선택할 객체 프롬프트" placeholder="사람 선택해 / red cup" value={text} onChange={e=>setText(e.target.value)} maxLength={120} disabled={disabled}/><button className="button primary full" disabled={disabled||!hasFrame||!text.trim()} type="submit"><Search size={13}/> 텍스트로 선택</button></form><div className="prompt-presets">{['사람 선택해','컵 선택해','의자 선택해'].map(prompt=><button key={prompt} disabled={disabled} onClick={()=>setText(prompt)}>{prompt.replace(' 선택해','')}</button>)}</div><p>현재 객체 목록을 새 결과로 교체합니다. 마스크를 확인한 뒤 선택 확정 · 추적을 누르세요.</p><small>한국어는 지원된 단어만 변환합니다. 복잡한 표현은 영어 명사구로 입력하세요. 현재 최대 4개 객체입니다.</small><p>SAM 3.1 실험 모드 · 첫 로딩과 추적에 시간이 걸릴 수 있습니다.</p></div>;
}
