"""
Visor 3D en el navegador (WebGL puro, sin librerias ni internet).

exportar_html(dist, campos, ruta, r=None) escribe un .html autocontenido.
Abrelo con doble clic (Chrome/Edge/Firefox):

  * raton izq. = girar | der. (o Mayus+izq.) = desplazar | rueda = zoom
  * campo de color: Lambda, distancia entre vecinos, error de densidad...
  * corte por un plano X/Y/Z: "semiespacio" (ves el interior del pajaro
    cortado) o "loncha" (una seccion de espesor elegido)
  * tamano de particula, superficie de la geometria en alambre
Las particulas se dibujan como esferas de diametro dx.
"""
import base64
import json
import numpy as np
import matplotlib

from .visor3d import campos_para_color


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _lut(nombre):
    cmap = matplotlib.colormaps[nombre]
    c = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return _b64(c, np.uint8)


def _aristas(F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def exportar_html(dist, campos, ruta, r=None, titulo=None):
    x = dist.x.astype(np.float64)
    centro = 0.5 * (dist.geom.bbox[0] + dist.geom.bbox[1])
    x = x - centro
    cc = campos_para_color(dist, campos)
    campos_js = []
    for nombre, (v, cmap, a, b) in cc.items():
        v = np.asarray(v, dtype=float)
        fin = np.isfinite(v)
        vmin = float(a) if a is not None else float(np.nanmin(v))
        vmax = float(b) if b is not None else float(np.nanmax(v))
        vv = np.where(fin, v, -1e30)
        campos_js.append({"nombre": nombre, "datos": _b64(vv, np.float32),
                          "vmin": vmin, "vmax": vmax, "lut": _lut(cmap)})
    malla = None
    try:
        V, F = dist.geom.malla(40)
        E = _aristas(F)
        malla = {"v": _b64(V - centro, np.float32), "e": _b64(E, np.uint32)}
    except NotImplementedError:
        pass
    datos = {
        "titulo": titulo or f"{dist.metodo} - {dist.geom.nombre}",
        "n": int(len(x)), "pos": _b64(x, np.float32), "dx": float(dist.dx_eq),
        "bbox": [(dist.geom.bbox[0] - centro).tolist(), (dist.geom.bbox[1] - centro).tolist()],
        "centro": centro.tolist(),
        "campos": campos_js, "malla": malla,
        "info": dist.resumen(),
        "metricas": ({k: (float(v) if isinstance(v, (int, float, np.integer, np.floating))
                          else str(v)) for k, v in r.items()} if r else {}),
    }
    html = _PLANTILLA.replace("/*__DATOS__*/null", json.dumps(datos))
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)
    return ruta


_PLANTILLA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visor SPH 3D</title>
<style>
 :root{--bg:#14161a;--panel:#1f2329;--txt:#e6e6e6;--sub:#9aa3ad;--acc:#4ea1ff}
 html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);font:13px system-ui,Segoe UI,Arial}
 #c{position:fixed;left:0;top:0;width:100%;height:100%;display:block}
 #panel{position:fixed;right:10px;top:10px;width:270px;max-height:calc(100% - 20px);overflow:auto;
   background:var(--panel);border-radius:8px;padding:10px 12px;box-shadow:0 2px 10px #0008}
 #panel h1{font-size:14px;margin:0 0 4px}
 .sub{color:var(--sub);font-size:11px;margin-bottom:8px}
 label{display:block;margin-top:8px;color:var(--sub);font-size:12px}
 select,input[type=range]{width:100%}
 select{background:#2a2f37;color:var(--txt);border:1px solid #3a404a;border-radius:4px;padding:3px}
 .fila{display:flex;gap:6px;align-items:center}
 .val{min-width:56px;text-align:right;font-variant-numeric:tabular-nums}
 #barra{width:100%;height:14px;border-radius:3px;margin-top:4px}
 .lim{display:flex;justify-content:space-between;font-size:11px;color:var(--sub)}
 details{margin-top:10px} pre{font-size:10.5px;white-space:pre-wrap;color:#cfd6de}
 #ayuda{position:fixed;left:10px;bottom:10px;color:var(--sub);font-size:11px;background:#0008;padding:6px 8px;border-radius:6px}
 #cont{position:fixed;left:10px;top:10px;color:var(--sub);font-size:12px;background:#0008;padding:6px 8px;border-radius:6px}
</style></head><body>
<canvas id="c"></canvas>
<div id="cont"></div>
<div id="panel">
 <h1 id="tit"></h1><div class="sub" id="inf"></div>
 <label>Campo de color</label><select id="campo"></select>
 <canvas id="barra" width="256" height="14"></canvas><div class="lim"><span id="vmin"></span><span id="vmax"></span></div>
 <label>Corte</label>
 <div class="fila"><select id="eje"><option value="-1">sin corte</option><option value="0" selected>X</option><option value="1">Y</option><option value="2">Z</option></select>
 <select id="modo"><option value="1">semiespacio (&le; corte)</option><option value="3">semiespacio (&ge; corte)</option><option value="2">loncha</option></select></div>
 <label>Posicion del corte</label><div class="fila"><input id="pos" type="range" min="0" max="1000" value="500"><span class="val" id="posv"></span></div>
 <label>Espesor de la loncha (en dx)</label><div class="fila"><input id="esp" type="range" min="0.5" max="8" step="0.25" value="1"><span class="val" id="espv"></span></div>
 <label>Tamano de particula (x dx)</label><div class="fila"><input id="tam" type="range" min="0.2" max="1.2" step="0.05" value="0.9"><span class="val" id="tamv"></span></div>
 <label><input type="checkbox" id="sup" checked> superficie de la geometria</label>
 <label><input type="checkbox" id="nan" checked> mostrar particulas sin valor (gris)</label>
 <details><summary>Metricas</summary><pre id="met"></pre></details>
</div>
<div id="ayuda">izq: girar &middot; der / Mayus+izq: mover &middot; rueda: zoom &middot; doble clic: reiniciar vista</div>
<script>
const D = /*__DATOS__*/null;
function dec(b64, T){const s=atob(b64);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer);}
const pos = dec(D.pos, Float32Array);
const cv = document.getElementById('c');
const gl = cv.getContext('webgl',{antialias:true});
if(!gl){document.body.innerHTML='<p style="padding:20px">Este navegador no tiene WebGL activado.</p>';}
// ---------- shaders
function sh(t,src){const s=gl.createShader(t);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(s);return s;}
function prog(v,f){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,v));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);return p;}
const P = prog(`
attribute vec3 aPos; attribute float aVal;
uniform mat4 uMV, uP; uniform float uSize, uScale, uVmin, uVmax, uCut, uHalf, uNan;
uniform vec3 uAx; uniform int uModo;
varying float vT; varying float vNan;
void main(){
  float c = dot(aPos,uAx); bool vis = true;
  if(uModo==1) vis = c<=uCut; else if(uModo==3) vis = c>=uCut; else if(uModo==2) vis = abs(c-uCut)<=uHalf;
  vNan = aVal < -1e29 ? 1.0 : 0.0;
  if(vNan>0.5 && uNan<0.5) vis=false;
  vec4 mv = uMV*vec4(aPos,1.0);
  gl_Position = vis ? uP*mv : vec4(2.0,2.0,2.0,1.0);
  gl_PointSize = vis ? uSize*uScale/(-mv.z) : 0.0;
  vT = clamp((aVal-uVmin)/max(uVmax-uVmin,1e-30),0.0,1.0);
}`,`
precision mediump float; uniform sampler2D uLut; varying float vT; varying float vNan;
void main(){
  vec2 q = gl_PointCoord*2.0-1.0; float r2 = dot(q,q); if(r2>1.0) discard;
  vec3 n = vec3(q.x,-q.y,sqrt(1.0-r2));
  vec3 base = vNan>0.5 ? vec3(0.55) : texture2D(uLut, vec2(vT,0.5)).rgb;
  float d = max(dot(n, normalize(vec3(0.4,0.5,0.8))),0.0);
  float s = pow(max(dot(n, normalize(vec3(0.2,0.3,1.0))),0.0),40.0);
  gl_FragColor = vec4(base*(0.35+0.65*d)+0.25*s,1.0);
}`);
const PL = prog(`attribute vec3 aPos; uniform mat4 uMV,uP; void main(){gl_Position=uP*uMV*vec4(aPos,1.0);}`,
 `precision mediump float; void main(){gl_FragColor=vec4(0.75,0.8,0.9,0.18);}`);
// ---------- buffers
const bPos = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,bPos); gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
const bVal = gl.createBuffer();
let malla=null;
if(D.malla){const v=dec(D.malla.v,Float32Array), e=dec(D.malla.e,Uint32Array);
  const L=new Float32Array(e.length*3); for(let i=0;i<e.length;i++){L.set(v.subarray(e[i]*3,e[i]*3+3),i*3);}
  malla={b:gl.createBuffer(),n:e.length}; gl.bindBuffer(gl.ARRAY_BUFFER,malla.b); gl.bufferData(gl.ARRAY_BUFFER,L,gl.STATIC_DRAW);}
const tex = gl.createTexture();
// ---------- estado
const S = {yaw:-0.7, pitch:0.45, dist:0, tx:0,ty:0,tz:0, campo:0};
const bb = D.bbox, ext=[bb[1][0]-bb[0][0],bb[1][1]-bb[0][1],bb[1][2]-bb[0][2]];
const R = 0.5*Math.hypot(ext[0],ext[1],ext[2]);
function reset(){S.yaw=-0.7;S.pitch=0.45;S.dist=R*3.0;S.tx=S.ty=S.tz=0;}
reset();
const $=id=>document.getElementById(id);
$('tit').textContent=D.titulo; $('inf').textContent=D.info;
$('met').textContent=Object.entries(D.metricas).map(([k,v])=>k.padEnd(20)+(typeof v==='number'?(+v.toPrecision(4)):v)).join('\n');
D.campos.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.nombre;$('campo').appendChild(o);});
function setCampo(i){
  S.campo=i; const c=D.campos[i];
  gl.bindBuffer(gl.ARRAY_BUFFER,bVal); gl.bufferData(gl.ARRAY_BUFFER,dec(c.datos,Float32Array),gl.STATIC_DRAW);
  const lut=dec(c.lut,Uint8Array); const rgba=new Uint8Array(256*4);
  for(let k=0;k<256;k++){rgba[4*k]=lut[3*k];rgba[4*k+1]=lut[3*k+1];rgba[4*k+2]=lut[3*k+2];rgba[4*k+3]=255;}
  gl.bindTexture(gl.TEXTURE_2D,tex); gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,256,1,0,gl.RGBA,gl.UNSIGNED_BYTE,rgba);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  const bc=$('barra').getContext('2d'); const im=bc.createImageData(256,14);
  for(let x=0;x<256;x++)for(let y=0;y<14;y++){const p=4*(y*256+x);im.data[p]=lut[3*x];im.data[p+1]=lut[3*x+1];im.data[p+2]=lut[3*x+2];im.data[p+3]=255;}
  bc.putImageData(im,0,0); $('vmin').textContent=(+c.vmin.toPrecision(3)); $('vmax').textContent=(+c.vmax.toPrecision(3));
  draw();
}
// ---------- matrices
function persp(f,a,n,fa){const t=1/Math.tan(f/2);return [t/a,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];}
function mul(a,b){const o=new Array(16).fill(0);for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)o[j*4+i]+=a[k*4+i]*b[j*4+k];return o;}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function nrm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function base(){ // vectores de camara (Z del mundo = arriba)
  const cp=Math.cos(S.pitch), d=[cp*Math.cos(S.yaw),cp*Math.sin(S.yaw),Math.sin(S.pitch)];
  const t=[S.tx,S.ty,S.tz], eye=[t[0]+S.dist*d[0],t[1]+S.dist*d[1],t[2]+S.dist*d[2]];
  const f=nrm(sub(t,eye)), r=nrm(cross(f,[0,0,1])), u=cross(r,f);
  return {eye,f,r,u};
}
function view(){
  const {eye,f,r,u}=base();
  return [r[0],u[0],-f[0],0, r[1],u[1],-f[1],0, r[2],u[2],-f[2],0, -dot(r,eye),-dot(u,eye),dot(f,eye),1];
}
// ---------- dibujo
function resize(){const d=window.devicePixelRatio||1;cv.width=innerWidth*d;cv.height=innerHeight*d;gl.viewport(0,0,cv.width,cv.height);draw();}
let pend=false; function draw(){if(!pend){pend=true;requestAnimationFrame(render);}}
function ctl(){
  const ax=+$('eje').value, modo=ax<0?0:+$('modo').value;
  const t=+$('pos').value/1000; let cut=0, half=0;
  if(ax>=0){cut=bb[0][ax]+t*ext[ax]; $('posv').textContent=(cut+D.centro[ax]).toFixed(1);} else $('posv').textContent='-';
  half=0.5*(+$('esp').value)*D.dx; $('espv').textContent=$('esp').value; $('tamv').textContent=$('tam').value;
  return {ax,modo,cut,half};
}
function render(){
  pend=false; const c=ctl();
  gl.clearColor(0.08,0.086,0.1,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
  const fov=0.6, Pm=persp(fov,cv.width/cv.height,R*0.02,R*20), MV=view();
  gl.useProgram(P);
  const u=n=>gl.getUniformLocation(P,n);
  gl.uniformMatrix4fv(u('uMV'),false,MV); gl.uniformMatrix4fv(u('uP'),false,Pm);
  gl.uniform1f(u('uSize'),D.dx*(+$('tam').value)); gl.uniform1f(u('uScale'),cv.height/(2*Math.tan(fov/2)));
  const cc=D.campos[S.campo]; gl.uniform1f(u('uVmin'),cc.vmin); gl.uniform1f(u('uVmax'),cc.vmax);
  gl.uniform1f(u('uCut'),c.cut); gl.uniform1f(u('uHalf'),c.half); gl.uniform1f(u('uNan'),$('nan').checked?1:0);
  gl.uniform3fv(u('uAx'),[c.ax==0?1:0,c.ax==1?1:0,c.ax==2?1:0]); gl.uniform1i(u('uModo'),c.modo);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,tex); gl.uniform1i(u('uLut'),0);
  const aP=gl.getAttribLocation(P,'aPos'), aV=gl.getAttribLocation(P,'aVal');
  gl.bindBuffer(gl.ARRAY_BUFFER,bPos); gl.enableVertexAttribArray(aP); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,bVal); gl.enableVertexAttribArray(aV); gl.vertexAttribPointer(aV,1,gl.FLOAT,false,0,0);
  gl.drawArrays(gl.POINTS,0,D.n);
  gl.disableVertexAttribArray(aV);
  if(malla && $('sup').checked){
    gl.useProgram(PL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
    gl.uniformMatrix4fv(gl.getUniformLocation(PL,'uMV'),false,MV); gl.uniformMatrix4fv(gl.getUniformLocation(PL,'uP'),false,Pm);
    const a=gl.getAttribLocation(PL,'aPos'); gl.bindBuffer(gl.ARRAY_BUFFER,malla.b); gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);
    gl.drawArrays(gl.LINES,0,malla.n); gl.depthMask(true); gl.disable(gl.BLEND);
  }
  let vis=0; if(c.modo==0) vis=D.n; else { for(let i=0;i<D.n;i++){const v=pos[3*i+c.ax]; if(c.modo==1? v<=c.cut : c.modo==3? v>=c.cut : Math.abs(v-c.cut)<=c.half) vis++;} }
  $('cont').textContent=`N = ${D.n}   visibles = ${vis}`;
}
// ---------- raton
let drag=null;
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,b:(e.button==2||e.shiftKey)?'pan':'rot'};});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
  if(drag.b=='rot'){S.yaw-=dx*0.008;S.pitch=Math.max(-1.55,Math.min(1.55,S.pitch+dy*0.008));}
  else{const k=S.dist*0.0012, b=base();S.tx-=(dx*b.r[0]-dy*b.u[0])*k;S.ty-=(dx*b.r[1]-dy*b.u[1])*k;S.tz-=(dx*b.r[2]-dy*b.u[2])*k;}
  draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();S.dist*=Math.exp(e.deltaY*0.001);draw();},{passive:false});
cv.addEventListener('dblclick',()=>{reset();draw();});
['eje','modo','pos','esp','tam','sup','nan'].forEach(id=>$(id).addEventListener('input',draw));
$('campo').addEventListener('change',e=>setCampo(+e.target.value));
addEventListener('resize',resize);
resize(); setCampo(0);
</script></body></html>
"""
