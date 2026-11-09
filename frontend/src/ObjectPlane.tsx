import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {Box,Focus,RotateCw} from 'lucide-react';
type Props={frame?:{id:number;width:number;height:number;jpeg:string};object?:{id:number;name:string;mask_png:string|null;bbox:number[]|null}};
export function ObjectPlane({frame,object}:Props){
  const host=useRef<HTMLDivElement>(null),reset=useRef(()=>{}),orbit=useRef<OrbitControls|null>(null);
  const mesh=useRef<THREE.Mesh<THREE.PlaneGeometry,THREE.MeshBasicMaterial>|null>(null);
  const [error,setError]=useState<string|null>(null),[spin,setSpin]=useState(false);
  useEffect(()=>{if(orbit.current)orbit.current.autoRotate=spin;},[spin]);
  useEffect(()=>{
    if(!host.current)return;
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});}catch{setError('WebGL을 시작할 수 없습니다.');return;}
    const container=host.current,scene=new THREE.Scene();scene.background=new THREE.Color('#0c141e');
    const camera=new THREE.PerspectiveCamera(42,1,.01,100);camera.position.set(0,.1,4.2);
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','선택 객체의 평면 이미지. 드래그로 3D 공간에서 회전할 수 있습니다.');
    const controls=new OrbitControls(camera,renderer.domElement);orbit.current=controls;controls.enableDamping=true;
    controls.minDistance=.4;controls.maxDistance=15;controls.autoRotateSpeed=1.1;
    const material=new THREE.MeshBasicMaterial({transparent:true,alphaTest:.03,side:THREE.DoubleSide,depthWrite:false});
    const plane=new THREE.Mesh(new THREE.PlaneGeometry(1,1),material);plane.visible=false;scene.add(plane);mesh.current=plane;
    const grid=new THREE.GridHelper(8,32,0x304a4e,0x192c3a);grid.position.y=-1.2;scene.add(grid);
    const axes=new THREE.AxesHelper(.45);axes.position.set(-1.6,-1.18,0);scene.add(axes);
    reset.current=()=>{camera.position.set(0,.1,4.2);controls.target.set(0,0,0);controls.update();};
    const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();};
    const observer=new ResizeObserver(resize);observer.observe(container);resize();let raf=0;
    function draw(){controls.update();renderer.render(scene,camera);raf=requestAnimationFrame(draw);}draw();
    return()=>{cancelAnimationFrame(raf);observer.disconnect();controls.dispose();orbit.current=null;mesh.current=null;
      material.map?.dispose();material.dispose();plane.geometry.dispose();grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();axes.geometry.dispose();(axes.material as THREE.Material).dispose();renderer.dispose();renderer.domElement.remove();};
  },[]);
  useEffect(()=>{
    let cancelled=false;
    const plane=mesh.current;
    if(!plane)return;
    if(!frame||!object?.mask_png||!object.bbox){plane.visible=false;return;}
    const load=async(src:string)=>{const image=new Image();image.src=src;await image.decode();return image;};
    const source=frame,selected=object;
    Promise.all([load('data:image/jpeg;base64,'+source.jpeg),load('data:image/png;base64,'+selected.mask_png)]).then(([photo,mask])=>{
      if(cancelled)return;
      const [x0,y0,x1,y1]=selected.bbox!;const w=x1-x0+1,h=y1-y0+1,scale=Math.min(1,1024/Math.max(w,h));
      const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));
      const ctx=canvas.getContext('2d')!;ctx.drawImage(photo,x0,y0,w,h,0,0,canvas.width,canvas.height);
      const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(mask,x0,y0,w,h,0,0,canvas.width,canvas.height);const alpha=ctx.getImageData(0,0,canvas.width,canvas.height);
      for(let i=0;i<pixels.data.length;i+=4)pixels.data[i+3]=alpha.data[i];
      ctx.putImageData(pixels,0,0);const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
      plane.material.map?.dispose();plane.material.map=texture;plane.material.needsUpdate=true;
      plane.scale.set(2*w/Math.max(w,h),2*h/Math.max(w,h),1);plane.visible=true;setError(null);
    }).catch(()=>{if(!cancelled){plane.visible=false;setError('선택 이미지와 마스크를 불러오지 못했습니다.');}});
    return()=>{cancelled=true;};
  },[frame?.id,object?.id,object?.mask_png]);
  return <div className="scan-viewer"><div ref={host} className="scan-canvas"/><div className="scan-viewer-label"><span className="local-dot"/>{object?object.name+' · 단일 이미지':'선택 객체 평면 미리보기'}</div>
    {(!object?.mask_png||!object.bbox)&&!error&&<div className="scan-placeholder"><Box size={38}/><strong>객체를 선택하면 바로 표시됩니다</strong><p>배경을 제거한 한 장의 이미지를<br/>3D 공간의 평면에 배치합니다.</p></div>}
    {error&&<div role="alert" className="scan-viewer-error">{error}</div>}
    <div className="scan-viewer-tools"><div>평면 이미지 · 두께 없음</div><button aria-label="평면 이미지 시점 초기화" onClick={()=>reset.current()}><Focus size={14}/></button><button aria-label="평면 이미지 자동 회전" aria-pressed={spin} onClick={()=>setSpin(!spin)}><RotateCw size={14}/></button></div>
    <div className="scan-viewer-guide">드래그 회전 · 휠 확대 · 우클릭 이동<span>실제 입체 복원은 3D 스캔 모드</span></div></div>;
}
