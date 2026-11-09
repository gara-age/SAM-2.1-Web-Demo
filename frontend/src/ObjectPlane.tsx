import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {Box,Focus,RotateCw} from 'lucide-react';
import {objectRelief} from './objectRelief';
export type PreviewFrame={id:number;width:number;height:number;jpeg:string};
export type PreviewObject={id:number;name:string;mask_png:string|null;bbox:number[]|null;area:number};
type Props={frame?:PreviewFrame;object?:PreviewObject;mode?:'volume'|'plane';depth?:number};
export function ObjectPlane({frame,object,mode='volume',depth=.28}:Props){
  const host=useRef<HTMLDivElement>(null),reset=useRef(()=>{}),orbit=useRef<OrbitControls|null>(null);
  const mesh=useRef<THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial[]>|null>(null);
  const [error,setError]=useState<string|null>(null),[spin,setSpin]=useState(false);
  useEffect(()=>{if(orbit.current)orbit.current.autoRotate=spin;},[spin]);
  useEffect(()=>{
    if(!host.current)return;
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});}catch{setError('WebGL을 시작할 수 없습니다.');return;}
    const container=host.current,scene=new THREE.Scene();scene.background=new THREE.Color('#0c141e');
    const camera=new THREE.PerspectiveCamera(42,1,.01,100);camera.position.set(.55,.15,4.2);
    renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','선택 객체 입체 미리보기. 드래그로 앞면과 옆면을 회전해 볼 수 있습니다.');
    const controls=new OrbitControls(camera,renderer.domElement);orbit.current=controls;controls.enableDamping=true;
    controls.minDistance=.4;controls.maxDistance=15;controls.autoRotateSpeed=1.1;
    scene.add(new THREE.HemisphereLight(0xe9f4ff,0x263b3e,2));
    const key=new THREE.DirectionalLight(0xffffff,2);key.position.set(-3,4,5);scene.add(key);
    const fill=new THREE.DirectionalLight(0x83b9c8,1);fill.position.set(3,1,-3);scene.add(fill);
    const front=new THREE.MeshStandardMaterial({transparent:true,alphaTest:.03,roughness:.88,side:THREE.FrontSide});
    const back=new THREE.MeshStandardMaterial({color:0x527c83,roughness:.9,side:THREE.FrontSide});
    const surface=new THREE.Mesh(new THREE.BufferGeometry(),[front,back]);surface.visible=false;scene.add(surface);mesh.current=surface;
    const grid=new THREE.GridHelper(8,32,0x304a4e,0x192c3a);grid.position.y=-1.2;scene.add(grid);
    const axes=new THREE.AxesHelper(.45);axes.position.set(-1.6,-1.18,0);scene.add(axes);
    reset.current=()=>{camera.position.set(.55,.15,4.2);controls.target.set(0,0,0);controls.update();};
    const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();};
    const observer=new ResizeObserver(resize);observer.observe(container);resize();let raf=0;
    function draw(){controls.update();renderer.render(scene,camera);raf=requestAnimationFrame(draw);}draw();
    return()=>{cancelAnimationFrame(raf);observer.disconnect();controls.dispose();orbit.current=null;mesh.current=null;
      front.map?.dispose();front.dispose();back.dispose();surface.geometry.dispose();grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();axes.geometry.dispose();(axes.material as THREE.Material).dispose();renderer.dispose();renderer.domElement.remove();};
  },[]);
  useEffect(()=>{
    let cancelled=false;const surface=mesh.current;if(!surface)return;
    if(!frame||!object?.mask_png||!object.bbox){surface.visible=false;return;}
    const load=async(src:string)=>{const image=new Image();image.src=src;await image.decode();return image;};
    const source=frame,selected=object;
    Promise.all([load('data:image/jpeg;base64,'+source.jpeg),load('data:image/png;base64,'+selected.mask_png)]).then(([photo,mask])=>{
      if(cancelled)return;
      const [x0,y0,x1,y1]=selected.bbox!;const w=x1-x0+1,h=y1-y0+1,scale=Math.min(1,512/Math.max(w,h));
      const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));
      const ctx=canvas.getContext('2d')!;ctx.drawImage(photo,x0,y0,w,h,0,0,canvas.width,canvas.height);
      const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(mask,x0,y0,w,h,0,0,canvas.width,canvas.height);const alpha=ctx.getImageData(0,0,canvas.width,canvas.height);
      for(let i=0;i<pixels.data.length;i+=4)pixels.data[i+3]=alpha.data[i];
      ctx.putImageData(pixels,0,0);const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
      surface.material[0].map?.dispose();surface.material[0].map=texture;
      surface.material[0].side=mode==='plane'?THREE.DoubleSide:THREE.FrontSide;surface.material[0].needsUpdate=true;
      surface.geometry.dispose();surface.geometry=mode==='volume'?objectRelief(pixels.data,canvas.width,canvas.height,depth):new THREE.PlaneGeometry(2*w/Math.max(w,h),2*h/Math.max(w,h));
      surface.visible=true;setError(null);
    }).catch(()=>{if(!cancelled){surface.visible=false;setError('선택 이미지와 마스크를 불러오지 못했습니다.');}});
    return()=>{cancelled=true;};
  },[frame?.id,object?.id,object?.mask_png,mode,depth]);
  return <div className="scan-viewer"><div ref={host} className="scan-canvas"/><div className="scan-viewer-label"><span className="local-dot"/>{object?object.name+' · 프레임 '+frame?.id:'선택 객체 미리보기'}</div>
    {(!object?.mask_png||!object.bbox)&&!error&&<div className="scan-placeholder"><Box size={38}/><strong>객체를 선택하면 바로 표시됩니다</strong><p>마스크에 곡면과 두께를 더한<br/>추정 입체 미리보기입니다.</p></div>}
    {error&&<div role="alert" className="scan-viewer-error">{error}</div>}
    <div className="scan-viewer-tools"><div>{mode==='volume'?'앞·뒤 곡면 + 옆면 · 추정 입체':'원본 평면 이미지'}</div><button aria-label="객체 미리보기 시점 초기화" onClick={()=>reset.current()}><Focus size={14}/></button><button aria-label="객체 미리보기 자동 회전" aria-pressed={spin} onClick={()=>setSpin(!spin)}><RotateCw size={14}/></button></div>
    <div className="scan-viewer-guide">드래그 회전 · 휠 확대<span>실제 치수·깊이 측정값 아님</span></div></div>;
}
