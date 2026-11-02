/** Map an SVG meet/contain viewport to original integer pixel indices. */
export function imagePoint(clientX:number,clientY:number,
  rect:{left:number;top:number;width:number;height:number},width:number,height:number):[number,number]|null {
  const scale=Math.min(rect.width/width,rect.height/height);
  if(!Number.isFinite(scale)||scale<=0)return null;
  const offsetX=(rect.width-width*scale)/2;
  const offsetY=(rect.height-height*scale)/2;
  const x=Math.floor((clientX-rect.left-offsetX)/scale);
  const y=Math.floor((clientY-rect.top-offsetY)/scale);
  return x>=0&&x<width&&y>=0&&y<height?[x,y]:null;
}
