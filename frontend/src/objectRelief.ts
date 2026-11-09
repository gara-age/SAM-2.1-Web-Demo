import * as THREE from 'three';

/** Closed silhouette-inflation mesh. Shape is an approximation, not measured depth. */
export function objectRelief(alpha:Uint8ClampedArray,width:number,height:number,depth=.28,resolution=96){
  const nx=Math.max(4,Math.round(resolution*width/Math.max(width,height))),ny=Math.max(4,Math.round(resolution*height/Math.max(width,height)));
  const count=nx*ny,inside=new Uint8Array(count),distance=new Float32Array(count);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const i=y*nx+x,px=Math.round(x/(nx-1)*(width-1)),py=Math.round(y/(ny-1)*(height-1));
    inside[i]=alpha[(py*width+px)*4+3]>=128?1:0;
    // A zero border ensures distance exists even for a rectangular full mask.
    distance[i]=inside[i]&&x>0&&y>0&&x<nx-1&&y<ny-1?1e6:0;
  }
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const i=y*nx+x;if(x)distance[i]=Math.min(distance[i],distance[i-1]+1);if(y)distance[i]=Math.min(distance[i],distance[i-nx]+1);
  }
  for(let y=ny-1;y>=0;y--)for(let x=nx-1;x>=0;x--){
    const i=y*nx+x;if(x<nx-1)distance[i]=Math.min(distance[i],distance[i+1]+1);if(y<ny-1)distance[i]=Math.min(distance[i],distance[i+nx]+1);
  }
  let maximum=1;for(const d of distance)maximum=Math.max(maximum,d);
  const positions=new Float32Array(count*6),uv=new Float32Array(count*4);
  const objectWidth=2*width/Math.max(width,height),objectHeight=2*height/Math.max(width,height);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const i=y*nx+x,u=x/(nx-1),v=y/(ny-1),bulge=.025+depth*Math.sqrt(distance[i]/maximum);
    positions.set([(u-.5)*objectWidth,(.5-v)*objectHeight,bulge],i*3);
    positions.set([(u-.5)*objectWidth,(.5-v)*objectHeight,-bulge*.7],(i+count)*3);
    uv.set([u,1-v],i*2);uv.set([u,1-v],(i+count)*2);
  }
  const front:number[]=[],back:number[]=[],sides:number[]=[],edges=new Map<string,{a:number;b:number;count:number}>();
  function triangle(a:number,b:number,c:number){
    front.push(a,b,c);back.push(a+count,c+count,b+count);
    for(const [start,end] of [[a,b],[b,c],[c,a]]){
      const key=Math.min(start,end)+':'+Math.max(start,end),edge=edges.get(key);
      if(edge)edge.count++;else edges.set(key,{a:start,b:end,count:1});
    }
  }
  for(let y=0;y<ny-1;y++)for(let x=0;x<nx-1;x++){
    const a=y*nx+x,b=a+1,c=a+nx,d=c+1;
    if(inside[a]&&inside[c]&&inside[b])triangle(a,c,b);
    if(inside[b]&&inside[c]&&inside[d])triangle(b,c,d);
  }
  for(const edge of edges.values())if(edge.count===1){const {a,b}=edge;sides.push(b,a,a+count,b,a+count,b+count);}
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('uv',new THREE.BufferAttribute(uv,2));
  geometry.setIndex([...front,...back,...sides]);geometry.addGroup(0,front.length,0);geometry.addGroup(front.length,back.length+sides.length,1);
  geometry.computeVertexNormals();geometry.computeBoundingBox();geometry.computeBoundingSphere();return geometry;
}
