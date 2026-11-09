import test from 'node:test';
import assert from 'node:assert/strict';
import {objectRelief} from './src/objectRelief.ts';

function silhouette(w:number,h:number){const rgba=new Uint8ClampedArray(w*h*4);for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(((x-w/2)/(w*.4))**2+((y-h/2)/(h*.42))**2<1)rgba[(y*w+x)*4+3]=255;return rgba;}
test('inflated silhouette is finite, has real depth, and is a closed indexed surface',()=>{
  const geometry=objectRelief(silhouette(48,64),48,64,.3,40),p=geometry.getAttribute('position'),indices=geometry.getIndex()!;
  assert(indices.count>0);assert(geometry.boundingBox!.max.z-geometry.boundingBox!.min.z>.3);
  for(const value of p.array)assert(Number.isFinite(value));
  const edges=new Map<string,number>();
  for(let i=0;i<indices.count;i+=3){const a=indices.getX(i),b=indices.getX(i+1),c=indices.getX(i+2);for(const [x,y] of [[a,b],[b,c],[c,a]]){const key=Math.min(x,y)+':'+Math.max(x,y);edges.set(key,(edges.get(key)??0)+1);}}
  for(const count of edges.values())assert.equal(count,2);
  assert.equal(geometry.groups.length,2);geometry.dispose();
});
test('thickness increases actual depth and UV keeps the photograph upright',()=>{
  const alpha=silhouette(48,64),thin=objectRelief(alpha,48,64,.1,32),thick=objectRelief(alpha,48,64,.5,32);
  assert(thick.boundingBox!.max.z>thin.boundingBox!.max.z);
  const p=thin.getAttribute('position'),uv=thin.getAttribute('uv');assert(p.getY(0)>0);assert.equal(uv.getY(0),1);
  thin.dispose();thick.dispose();
});
