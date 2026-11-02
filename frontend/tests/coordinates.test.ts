import {test} from 'node:test';
import {deepStrictEqual,strictEqual} from 'node:assert';
import {imagePoint} from '../src/coordinates.ts';

test('scaled 4:3 center and inclusive final pixel',()=>{
  const rect={left:10,top:20,width:320,height:240};
  deepStrictEqual(imagePoint(170,140,rect,640,480),[320,240]);
  deepStrictEqual(imagePoint(329.9,259.9,rect,640,480),[639,479]);
  strictEqual(imagePoint(330,260,rect,640,480),null);
});
test('16:9 video with top and bottom letterbox in 4:3 viewport',()=>{
  const rect={left:100,top:100,width:640,height:480};
  deepStrictEqual(imagePoint(420,340,rect,1280,720),[640,360]);
  deepStrictEqual(imagePoint(100,160,rect,1280,720),[0,0]);
  strictEqual(imagePoint(420,159,rect,1280,720),null);
  strictEqual(imagePoint(420,520,rect,1280,720),null);
});
test('portrait video with side letterbox',()=>{
  const rect={left:0,top:0,width:640,height:480};
  deepStrictEqual(imagePoint(320,240,rect,480,640),[240,320]);
  strictEqual(imagePoint(139,240,rect,480,640),null);
});
test('zero-size viewport is not clickable',()=>{
  strictEqual(imagePoint(0,0,{left:0,top:0,width:0,height:0},640,480),null);
});
