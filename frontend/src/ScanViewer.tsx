import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {PLYLoader} from 'three/addons/loaders/PLYLoader.js';
import {Box,Focus,Maximize,RotateCw} from 'lucide-react';

type Props={url:string|null;version:number;label:string;aligned?:boolean};

export function ScanViewer({url,version,label,aligned}:Props){
  const host=useRef<HTMLDivElement>(null),wrapper=useRef<HTMLDivElement>(null);
  const controls=useRef<OrbitControls|null>(null),reset=useRef<()=>void>(()=>{});
  const [mode,setMode]=useState<'surface'|'wire'|'points'>('surface');
  const [spin,setSpin]=useState(false),[error,setError]=useState<string|null>(null),[loading,setLoading]=useState(false);
  const [flipped,setFlipped]=useState(false);
  useEffect(()=>setFlipped(false),[url,version]);
  useEffect(()=>{if(controls.current)controls.current.autoRotate=spin;},[spin]);
  useEffect(()=>{
    if(!host.current)return;
    const container=host.current,abort=new AbortController();
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});}
    catch{setError('WebGL을 시작할 수 없습니다. 브라우저의 GPU 지원을 확인하세요.');return;}
    let stopped=false,raf=0;
    const scene=new THREE.Scene();scene.background=new THREE.Color('#0c141e');
    const camera=new THREE.PerspectiveCamera(42,1,.01,100);
    camera.position.set(.1,.15,4.2);
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-label','3D 스캔 모델. 드래그로 회전하고 휠로 확대할 수 있습니다.');
    const orbit=new OrbitControls(camera,renderer.domElement);controls.current=orbit;
    orbit.enableDamping=true;orbit.autoRotate=spin;orbit.autoRotateSpeed=1.4;orbit.minDistance=.2;orbit.maxDistance=15;
    scene.add(new THREE.HemisphereLight(0xdcefff,0x2a3a40,2.2));
    const key=new THREE.DirectionalLight(0xffffff,2.2);key.position.set(3,6,4);scene.add(key);
    const fill=new THREE.DirectionalLight(0x8abfff,1.3);fill.position.set(-4,1,-2);scene.add(fill);
    const grid=new THREE.GridHelper(8,32,0x304a4e,0x192c3a);grid.position.y=-1.35;scene.add(grid);
    const axes=new THREE.AxesHelper(.45);axes.position.set(-1.6,-1.33,0);scene.add(axes);
    reset.current=()=>{camera.position.set(.1,.15,4.2);orbit.target.set(0,0,0);orbit.update();};
    const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();};
    const observer=new ResizeObserver(resize);observer.observe(container);resize();
    function draw(){if(stopped)return;orbit.update();renderer.render(scene,camera);raf=requestAnimationFrame(draw);}draw();
    setError(null);
    if(url){
      setLoading(true);
      const asset=(mode==='points'?url.replace('model.ply','points.ply'):url)+`?v=${version}`;
      fetch(asset,{signal:abort.signal}).then(r=>{if(!r.ok)throw new Error('3D 모델 파일을 불러올 수 없습니다.');return r.arrayBuffer();}).then(buffer=>{
        if(stopped)return;
        const geometry=new PLYLoader().parse(buffer);
        if(!geometry.getAttribute('position')?.count)throw new Error('표시할 3D 점이 없습니다.');
        geometry.computeBoundingBox();
        const size=new THREE.Vector3(),center=new THREE.Vector3();
        geometry.boundingBox!.getSize(size);geometry.boundingBox!.getCenter(center);
        const extent=Math.max(size.x,size.y,size.z);
        if(!Number.isFinite(extent)||extent<=0){geometry.dispose();throw new Error('올바르지 않은 3D 모델 크기입니다.');}
        geometry.translate(-center.x,-center.y,-center.z);geometry.scale(2/extent,2/extent,2/extent);
        if(flipped)geometry.rotateZ(Math.PI);
        const hasColor=!!geometry.getAttribute('color');
        if(mode==='points'){
          scene.add(new THREE.Points(geometry,new THREE.PointsMaterial({size:.014,vertexColors:hasColor,color:hasColor?0xffffff:0x6fdcbb,sizeAttenuation:true})));
        }else{
          geometry.computeVertexNormals();
          scene.add(new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:hasColor?0xffffff:0x9acabc,vertexColors:hasColor,roughness:.8,metalness:.05,side:THREE.DoubleSide,wireframe:mode==='wire'})));
        }
        setLoading(false);
      }).catch(e=>{if(!stopped&&!abort.signal.aborted){setError(String(e.message??e));setLoading(false);}});
    }else setLoading(false);
    return()=>{stopped=true;abort.abort();cancelAnimationFrame(raf);observer.disconnect();orbit.dispose();controls.current=null;
      scene.traverse(object=>{if(object instanceof THREE.Mesh||object instanceof THREE.Points||object instanceof THREE.LineSegments){object.geometry.dispose();const material=object.material;for(const item of Array.isArray(material)?material:[material])item.dispose();}});
      renderer.dispose();renderer.domElement.remove();reset.current=()=>{};
    };
  },[url,version,mode,flipped]);
  return <div className="scan-viewer" ref={wrapper}><div ref={host} className="scan-canvas"/>
    <div className="scan-viewer-label"><span className="local-dot"/>{url?label:'3D RECONSTRUCTION VIEW'}</div>
    {!url&&!error&&<div className="scan-placeholder"><div><Box size={38} strokeWidth={1}/></div><strong>대상을 입체로 기록하세요</strong><p>여러 각도의 사진을 모아 복원하면<br/>이곳에서 모델을 회전해 볼 수 있습니다.</p></div>}
    {loading&&<div className="scan-loading">3D 모델 불러오는 중…</div>}
    {error&&<div className="scan-viewer-error" role="alert">{error}</div>}
    <div className="scan-viewer-tools"><div>{(['surface','wire','points'] as const).map(value=><button key={value} className={mode===value?'active':''} onClick={()=>setMode(value)}>{value==='surface'?'표면':value==='wire'?'와이어':'점군'}</button>)}</div><button title="시점 초기화" aria-label="3D 시점 초기화" onClick={()=>reset.current()}><Focus size={14}/></button><button title="자동 회전" aria-label="3D 자동 회전" aria-pressed={spin} className={spin?'active':''} onClick={()=>setSpin(!spin)}><RotateCw size={14}/></button><button title="전체 화면" aria-label="3D 전체 화면" onClick={()=>wrapper.current?.requestFullscreen().catch(()=>setError('전체 화면을 사용할 수 없습니다.'))}><Maximize size={13}/></button></div>
    <button className="scan-flip" title="모델을 화면에서 180° 회전합니다. 내려받는 파일은 바뀌지 않습니다." aria-pressed={flipped} disabled={!url} onClick={()=>setFlipped(!flipped)}>상하 180°</button>
    <div className="scan-viewer-guide">드래그 회전 · 휠 확대 · 우클릭 이동<span>{aligned?'사진 위쪽 기준 보정 · 상대 크기':'상대 크기'}</span></div>
  </div>;
}
