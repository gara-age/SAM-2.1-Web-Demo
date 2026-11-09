import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {Focus,Pause,Play} from 'lucide-react';

type Cloud={frame_id:number;version:number;object_id:number;count:number;positions:string;colors:string;processing_ms:number;timestamp:number};
type Reply={state:string;error:string|null;result:Cloud|null;frame_id:number|null};
const bytes=(value:string)=>Uint8Array.from(atob(value),c=>c.charCodeAt(0));
export function DepthCloud({objectId,available}:{objectId?:number;available:boolean}){
  const host=useRef<HTMLDivElement>(null),points=useRef<THREE.Points|null>(null),reset=useRef(()=>{});
  const [reply,setReply]=useState<Reply|null>(null),[cloud,setCloud]=useState<Cloud|null>(null),[playing,setPlaying]=useState(true),[size,setSize]=useState(.012),[error,setError]=useState<string|null>(null);
  useEffect(()=>{
    const element=host.current;if(!element)return;
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});}catch{setError('WebGL을 사용할 수 없습니다.');return;}
    renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;element.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-label','실시간 깊이 점군. 드래그로 표면을 회전하고 휠로 확대합니다.');
    const scene=new THREE.Scene();scene.background=new THREE.Color('#0c141e');
    const camera=new THREE.PerspectiveCamera(42,1,.01,100);camera.position.set(.75,.15,3.8);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.2;controls.maxDistance=15;
    reset.current=()=>{camera.position.set(.75,.15,3.8);controls.target.set(0,0,0);controls.update();};
    const material=new THREE.PointsMaterial({size:.012,vertexColors:true,sizeAttenuation:true});
    const surface=new THREE.Points(new THREE.BufferGeometry(),material);scene.add(surface);points.current=surface;
    const grid=new THREE.GridHelper(8,32,0x304a4e,0x192c3a);grid.position.y=-1.15;scene.add(grid);
    const resize=()=>{const w=element.clientWidth,h=element.clientHeight;if(w&&h){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}};
    const observer=new ResizeObserver(resize);observer.observe(element);resize();let raf=0;
    const draw=()=>{controls.update();renderer.render(scene,camera);raf=requestAnimationFrame(draw);};draw();
    return()=>{cancelAnimationFrame(raf);observer.disconnect();controls.dispose();surface.geometry.dispose();material.dispose();grid.geometry.dispose();(grid.material as THREE.Material).dispose();renderer.dispose();renderer.domElement.remove();points.current=null;};
  },[]);
  useEffect(()=>{if(points.current)(points.current.material as THREE.PointsMaterial).size=size;},[size]);
  useEffect(()=>{
    setCloud(null);setReply(null);if(points.current)points.current.visible=false;
  },[objectId,available]);
  useEffect(()=>{
    let cancelled=false,timer=0,since:number|undefined;const controller=new AbortController();
    if(!objectId||!available||!playing)return;
    async function poll(){
      try{
        const response=await fetch('/api/depth/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({object_id:objectId,since}),signal:controller.signal});
        if(!response.ok)throw new Error('깊이 처리 서버에 연결하지 못했습니다.');
        const next:Reply=await response.json();if(cancelled)return;
        setReply(next);setError(null);
        if(next.result){since=next.result.version;setCloud(next.result);}
        else if(next.frame_id===null){setCloud(null);if(points.current)points.current.visible=false;}
      }catch(reason){if(!cancelled)setError(reason instanceof Error?reason.message:String(reason));}
      if(!cancelled)timer=window.setTimeout(poll,250);
    }
    void poll();return()=>{cancelled=true;controller.abort();clearTimeout(timer);};
  },[objectId,available,playing]);
  useEffect(()=>{
    const surface=points.current;if(!surface||!cloud)return;
    const positions=bytes(cloud.positions),rgb=bytes(cloud.colors),colors=new Float32Array(rgb.length),color=new THREE.Color();
    for(let i=0;i<rgb.length;i+=3){color.setRGB(rgb[i]/255,rgb[i+1]/255,rgb[i+2]/255,THREE.SRGBColorSpace);colors.set([color.r,color.g,color.b],i);}
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(positions.buffer),3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));geometry.computeBoundingSphere();
    surface.geometry.dispose();surface.geometry=geometry;surface.visible=true;
  },[cloud]);
  const message=reply?.error||error;
  return <><div className="scan-viewer"><div ref={host} className="scan-canvas"/><div className="scan-viewer-label"><span className="local-dot"/>실시간 깊이 · {cloud?`${cloud.count.toLocaleString()}점 · 프레임 ${cloud.frame_id}`:'선택 객체'}</div>
    {!cloud&&!message&&<div className="scan-placeholder"><strong>{!available?'객체를 선택하면 깊이 점군이 표시됩니다':!playing?'깊이 갱신 일시정지':reply?.state==='loading'?'깊이 모델 불러오는 중':'현재 프레임의 깊이를 계산하고 있습니다'}</strong><p>카메라에 보이는 표면만 복원합니다.</p></div>}
    {message&&<div className="scan-viewer-error" role="alert">{message}</div>}
    <div className="scan-viewer-tools"><div>AI 상대 깊이 · 컬러 점군</div><button aria-label="깊이 점군 시점 초기화" onClick={()=>reset.current()}><Focus size={14}/></button></div><div className="scan-viewer-guide">드래그 회전 · 휠 확대<span>상대 크기 · 보이지 않는 뒷면 없음</span></div></div>
    <div className="space-preview-controls"><button className="text-button" onClick={()=>setPlaying(!playing)}>{playing?<Pause size={12}/>:<Play size={12}/>} {playing?'깊이 갱신 일시정지':'깊이 갱신 재개'}</button><label>점 크기 <input aria-label="점군 점 크기" type="range" min="0.004" max="0.035" step="0.001" value={size} onChange={e=>setSize(Number(e.target.value))}/></label><span>{cloud?`계산 ${cloud.processing_ms.toFixed(0)}ms · ${playing?Math.max(0,(Date.now()/1000-cloud.timestamp)).toFixed(1)+'초 전 프레임':'갱신 정지'}`:'최신 프레임 우선 · 최대 5회/초'}</span></div>
    <p className="space-note">영상에서 AI가 추정한 상대 깊이와 같은 프레임의 SAM 마스크로 만듭니다. 움직임에 맞춰 현재 표면만 갱신하며 누적 스캔하지 않습니다. 실측 거리·치수와 뒷면은 제공하지 않습니다. GPU 부하에 따라 갱신이 느려질 수 있습니다.</p></>;
}
