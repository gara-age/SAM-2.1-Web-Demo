import {useEffect,useRef,useState} from 'react';
import {Download,Film,RefreshCw} from 'lucide-react';
import './recorded-video.css';

export type Recording={id:string;model:string;state:string;input_name:string;frame_count:number;expected_frames:number;fps:number;duration:number;first_source_index:number;video_url:string|null;created_at:number;error:string|null};
const labels:Record<string,string>={recording:'순차 분석 중',paused:'분석 일시정지',finalizing:'재생 영상 마무리 중',completed:'분석 완료',interrupted:'중단된 분석 구간',failed:'영상 저장 실패'};
export function RecordedVideo({current}:{current?:Recording|null}){
  const [items,setItems]=useState<Recording[]>([]),[selected,setSelected]=useState(''),[error,setError]=useState<string|null>(null);
  const autoSelected=useRef(''),player=useRef<HTMLVideoElement>(null);
  async function refresh(){try{const response=await fetch('/api/recordings');if(!response.ok)throw new Error('결과 목록을 불러오지 못했습니다.');const data:Recording[]=await response.json();setItems(data);setError(null);setSelected(old=>old||data.find(r=>r.video_url)?.id||'');}catch(e){setError(e instanceof Error?e.message:String(e));}}
  useEffect(()=>{void refresh();const timer=setInterval(()=>void refresh(),5000);return()=>clearInterval(timer);},[]);
  useEffect(()=>{void refresh();if(current?.video_url&&current.id!==autoSelected.current){autoSelected.current=current.id;setSelected(current.id);}},[current?.id,current?.state]);
  const result=items.find(r=>r.id===selected)??(current?.id===selected?current:null);
  return <section className="recorded-video panel" aria-label="순차 분석 결과 영상">
    <div className="panel-heading"><Film size={16}/><h2>순차 분석 결과 영상</h2><button className="text-button" aria-label="결과 영상 목록 새로고침" onClick={()=>void refresh()}><RefreshCw size={13}/></button></div>
    <div className="recorded-body">
      {current&&<div className={`record-progress ${current.state==='failed'?'failed':''}`} role="status"><strong>{current.model.toUpperCase()} · {labels[current.state]??current.state}</strong><span>{current.frame_count.toLocaleString()} / {current.expected_frames||'?'} 프레임</span>{current.expected_frames>0&&<progress value={current.frame_count} max={current.expected_frames}/>}{current.error&&<p>{current.error}</p>}</div>}
      <label htmlFor="recorded-select">저장된 분석 결과</label><select id="recorded-select" aria-label="저장된 분석 결과 영상" value={selected} onChange={e=>{setSelected(e.target.value);setError(null);}}><option value="">결과 영상 선택</option>{items.filter(r=>r.video_url).map(r=><option key={r.id} value={r.id}>{r.model.toUpperCase()} · {r.input_name} · {labels[r.state]??r.state} · {new Date(r.created_at*1000).toLocaleTimeString('ko-KR',{hour12:false})}</option>)}</select>
      {result?.video_url?<><video ref={player} key={result.id} controls playsInline preload="metadata" src={result.video_url} aria-label="마스크 포함 분석 결과 영상" onError={()=>setError('브라우저에서 영상을 읽지 못했습니다. MP4를 내려받아 확인하세요.')}/><div className="recorded-controls"><span>{result.model.toUpperCase()} · {result.frame_count}프레임 · {result.duration.toFixed(2)}초</span><select aria-label="결과 영상 재생 속도" defaultValue="1" key={result.id} onChange={e=>{if(player.current)player.current.playbackRate=Number(e.target.value);}}>{[.5,1,1.5,2].map(rate=><option key={rate} value={rate}>{rate}×</option>)}</select><a className="button" href={result.video_url+'?download=true'} download><Download size={13}/> MP4 다운로드</a></div>{result.state==='interrupted'&&<p>분석이 중단되기 전까지 저장한 구간입니다. 전체 분석 완료 결과가 아닙니다.</p>}{result.first_source_index>0&&<p>원본 {result.first_source_index+1}번째 프레임부터 분석한 구간입니다.</p>}</>:<p>동영상에서 SAM 순차 분석을 시작하면 프레임마다 마스크를 저장합니다. 분석이 끝나면 이곳에서 원본 FPS로 재생하고 탐색할 수 있습니다.</p>}
      <p>마스크·객체 번호가 영상에 포함됩니다. 원본 음성은 포함하지 않습니다. 결과는 D드라이브에 보관됩니다.</p>{error&&<p role="alert">{error}</p>}
    </div>
  </section>;
}
