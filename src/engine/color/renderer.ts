/**
 * WebGL2 preview renderer: draws a video/image source through the color
 * pipeline with the clip's spatial transform, reframe crop and mask applied.
 * Layers can go straight to the canvas or into an off-screen target (transitions / effects, see
 * ./compositor.ts); two frame-texture slots let two clips be drawn in one frame.
 */
import type { BBox, ClipMask, ColorGrade, HslChannel } from '@/types/project';
import { ADJUST_SCALE, HSL_CHANNELS } from '../defaults';
import { bakeCurves } from './curves';
import type { Lut3D } from './lut';
import { FRAGMENT_SHADER, VERTEX_SHADER } from './shaders';

export type FrameSource = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageBitmap;

export interface LayerParams {
  grade: ColorGrade;
  /** normalised centre offset [-1,1] */
  position: [number, number];
  scale: number;
  rotationDeg: number;
  opacity: number;
  blurPx: number;
  /** crop in source pixels (from the reframe engine) */
  crop: BBox | null;
  mask: ClipMask | null;
  maskRect: [number, number, number, number] | null;
  sourceWidth: number;
  sourceHeight: number;
  timeMs: number;
  /** video fade factor (1 = none): the picture is multiplied by it (fades from / to black) */
  fade?: number;
}

/** An off-screen colour target (texture + framebuffer). */
export interface RenderTarget {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  w: number;
  h: number;
}

export interface LayerOptions {
  /** frame-texture slot (0 or 1): two clips in one frame need both */
  slot?: number;
  /** draw into this target instead of the canvas */
  target?: RenderTarget | null;
  /** clear the target to opaque black first (default true for targets) */
  clear?: boolean;
  /** draw this source instead of the one given to setSource */
  source?: FrameSource;
}

const MASK_MODE: Record<ClipMask['shape'], number> = {
  rectangle: 1,
  circle: 2,
  split: 3,
  filmstrip: 4,
};

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error('createShader failed');
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`Shader compile error: ${log}`);
  }
  return sh;
}

export class ColorRenderer {
  private program: WebGLProgram;
  private uniforms = new Map<string, WebGLUniformLocation | null>();
  private frameTex: WebGLTexture[];
  private curvesTex: WebGLTexture;
  private lutTex: WebGLTexture;
  /** up to two LUTs stay on the GPU (a transition between two differently graded clips) */
  private lutSlots: Array<{ lut: Lut3D; tex: WebGLTexture; size: number }> = [];
  private lutSize = 2;
  private hasLut = false;
  private curvesKey = '';
  private curvesRef: ColorGrade['curves'] | null = null;
  private source: FrameSource | null = null;
  private vao: WebGLVertexArrayObject;
  readonly gl: WebGL2RenderingContext;

  constructor(public canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: false });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;
    const vs = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    const program = gl.createProgram();
    if (!program) throw new Error('createProgram failed');
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`);
    }
    this.program = program;

    const vao = gl.createVertexArray();
    if (!vao) throw new Error('createVertexArray failed');
    this.vao = vao;
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.frameTex = [this.createTex2D(), this.createTex2D()];
    this.curvesTex = this.createTex2D();
    this.lutTex = this.createTex3D();
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, 2, 2, 2, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array(24));

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  private createTex2D(): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture();
    if (!t) throw new Error('createTexture failed');
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private createTex3D(): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture();
    if (!t) throw new Error('createTexture failed');
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, t);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    return t;
  }

  private u(name: string): WebGLUniformLocation | null {
    if (!this.uniforms.has(name)) this.uniforms.set(name, this.gl.getUniformLocation(this.program, name));
    return this.uniforms.get(name) ?? null;
  }

  setSource(src: FrameSource | null): void {
    this.source = src;
  }

  /**
   * Select a 3D LUT; uploads only when it is not one of the two LUTs kept on the GPU (so a cut or a
   * transition between two graded clips costs nothing after the first frame).
   */
  setLut(lut: Lut3D | null): void {
    const gl = this.gl;
    if (!lut) {
      this.hasLut = false;
      return;
    }
    let slot = this.lutSlots.find((x) => x.lut === lut);
    if (!slot) {
      const reuse = this.lutSlots.length >= 2 ? this.lutSlots.shift()! : null;
      const tex = reuse?.tex ?? (this.lutSlots.length === 0 && !reuse ? this.lutTex : this.createTex3D());
      const s = lut.size;
      const bytes = new Uint8Array(s * s * s * 3);
      for (let i = 0; i < bytes.length; i++) {
        bytes[i] = Math.max(0, Math.min(255, Math.round(lut.data[i] * 255)));
      }
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, s, s, s, 0, gl.RGB, gl.UNSIGNED_BYTE, bytes);
      slot = { lut, tex, size: s };
      this.lutSlots.push(slot);
    } else {
      // most recently used last
      this.lutSlots = [...this.lutSlots.filter((x) => x !== slot), slot];
    }
    this.lutTex = slot.tex;
    this.lutSize = slot.size;
    this.hasLut = true;
  }

  private uploadCurves(grade: ColorGrade): void {
    // curves are immutable document values: compare by reference, fall back to content
    if (grade.curves === this.curvesRef) return;
    this.curvesRef = grade.curves;
    const key = JSON.stringify(grade.curves);
    if (key === this.curvesKey) return;
    this.curvesKey = key;
    const gl = this.gl;
    const px = bakeCurves(grade.curves, 256);
    // bind on the curves unit: binding on the active unit (0) would replace the frame texture
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.curvesTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 4, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
  }

  clear(target: RenderTarget | null = null): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target?.fbo ?? null);
    gl.viewport(0, 0, target?.w ?? this.canvas.width, target?.h ?? this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (target) gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Compute the uv transform: quad uv (0..1 over the canvas) -> source uv.
   * Fits the (cropped) source into the canvas with letterboxing, then applies
   * the clip transform (scale/position/rotation) about the canvas centre.
   */
  private uvMatrix(p: LayerParams): Float32Array {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const W = Math.max(1, p.sourceWidth);
    const H = Math.max(1, p.sourceHeight);
    const crop = p.crop ?? [0, 0, W, H];
    const cropW = Math.max(1, crop[2] - crop[0]);
    const cropH = Math.max(1, crop[3] - crop[1]);
    const srcAspect = cropW / cropH;
    const ax = cw / ch;
    // fitted size of the crop window in canvas-normalised units (1 = full canvas)
    let fitW = 1;
    let fitH = 1;
    if (srcAspect > ax) fitH = ax / srcAspect;
    else fitW = srcAspect / ax;
    fitW *= Math.max(1e-4, p.scale);
    fitH *= Math.max(1e-4, p.scale);
    const cx = 0.5 + p.position[0] * 0.5;
    const cy = 0.5 - p.position[1] * 0.5; // canvas v axis points up
    const rad = (p.rotationDeg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    // Inverse mapping canvas uv -> crop uv, derived from the forward transform:
    //   X = (c.x-0.5)*fitW*ax ; Y = (c.y-0.5)*fitH ; rotate ; u = cx + X'/ax ; v = cy + Y'
    // => X' = (u-cx)*ax ; Y' = v-cy ; X = X'cos + Y'sin ; Y = -X'sin + Y'cos
    //    c.x = X/(fitW*ax) + 0.5 ; c.y = Y/fitH + 0.5
    const a00 = cos / fitW; // du -> c.x
    const a01 = sin / (fitW * ax); // dv -> c.x
    const a10 = (-sin * ax) / fitH; // du -> c.y
    const a11 = cos / fitH; // dv -> c.y
    const t0 = 0.5 - a00 * cx - a01 * cy;
    const t1 = 0.5 - a10 * cx - a11 * cy;

    // crop uv -> source uv. Crop is in top-left image pixels, texture v is bottom-up.
    const sx = cropW / W;
    const sy = cropH / H;
    const ox = crop[0] / W;
    const oy = 1 - crop[3] / H;

    // column-major mat3: [m00 m10 m20, m01 m11 m21, m02 m12 m22]
    return new Float32Array([
      sx * a00,
      sy * a10,
      0,
      sx * a01,
      sy * a11,
      0,
      sx * t0 + ox,
      sy * t1 + oy,
      1,
    ]);
  }

  render(p: LayerParams, lut: Lut3D | null | undefined = undefined, opts: LayerOptions = {}): void {
    const gl = this.gl;
    const src = opts.source ?? this.source;
    const target = opts.target ?? null;
    if (!src) {
      this.clear(target);
      return;
    }
    if (lut !== undefined) this.setLut(lut);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target?.fbo ?? null);
    gl.viewport(0, 0, target?.w ?? this.canvas.width, target?.h ?? this.canvas.height);
    if (target && opts.clear !== false) {
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex[opts.slot === 1 ? 1 : 0]);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src as TexImageSource);
    } catch {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      this.clear(target);
      return;
    }
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.uniform1i(this.u('u_frame'), 0);

    this.uploadCurves(p.grade);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.curvesTex);
    gl.uniform1i(this.u('u_curves'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.uniform1i(this.u('u_lut'), 2);
    gl.uniform1f(this.u('u_useLut'), this.hasLut && p.grade.lutAssetId ? 1 : 0);
    gl.uniform1f(this.u('u_lutSize'), this.lutSize);
    gl.uniform1f(this.u('u_lutIntensity'), p.grade.lutIntensity);

    gl.uniform2f(this.u('u_texel'), 1 / Math.max(1, p.sourceWidth), 1 / Math.max(1, p.sourceHeight));
    gl.uniformMatrix3fv(this.u('u_uvTransform'), false, this.uvMatrix(p));

    const g = p.grade;
    // adjust sliders use CapCut's scale (-50..50, sharpness 0..50); HSL uses -100..100
    const a = (v: number) => (v ?? 0) / ADJUST_SCALE;
    const n = (v: number) => v / 100;
    gl.uniform1f(this.u('u_exposure'), a(g.exposure) * 3);
    gl.uniform1f(this.u('u_contrast'), a(g.contrast));
    gl.uniform1f(this.u('u_brightness'), a(g.brightness));
    gl.uniform1f(this.u('u_highlights'), a(g.highlights));
    gl.uniform1f(this.u('u_shadows'), a(g.shadows));
    gl.uniform1f(this.u('u_brilliance'), a(g.brilliance));
    gl.uniform1f(this.u('u_saturation'), a(g.saturation));
    gl.uniform1f(this.u('u_vibrance'), a(g.vibrance));
    gl.uniform1f(this.u('u_sharpness'), Math.max(0, a(g.sharpness)));
    gl.uniform1f(this.u('u_temperature'), a(g.temperature));
    gl.uniform1f(this.u('u_tint'), a(g.tint));
    gl.uniform3fv(this.u('u_lift'), g.lift);
    gl.uniform3fv(this.u('u_gamma'), g.gamma);
    gl.uniform3fv(this.u('u_gain'), g.gain);
    gl.uniform3fv(this.u('u_offset'), g.offset);
    const hsl = new Float32Array(24);
    HSL_CHANNELS.forEach((ch: HslChannel, i) => {
      const o = g.hsl[ch] ?? { h: 0, s: 0, l: 0 };
      hsl[i * 3] = n(o.h);
      hsl[i * 3 + 1] = n(o.s);
      hsl[i * 3 + 2] = n(o.l);
    });
    gl.uniform3fv(this.u('u_hsl[0]'), hsl);
    gl.uniform1f(this.u('u_vignette'), g.vignette);
    gl.uniform1f(this.u('u_grain'), g.grain);
    gl.uniform1f(this.u('u_time'), p.timeMs);
    gl.uniform1f(this.u('u_opacity'), p.opacity);
    gl.uniform1f(this.u('u_fade'), p.fade ?? 1);
    gl.uniform1f(this.u('u_blur'), p.blurPx);

    if (p.mask && p.maskRect) {
      gl.uniform1i(this.u('u_maskMode'), MASK_MODE[p.mask.shape]);
      // mask rect is top-left normalised; shader works in bottom-up canvas uv
      const [mx, my, mw, mh] = p.maskRect;
      gl.uniform4f(this.u('u_maskRect'), mx, 1 - my - mh, mw, mh);
      gl.uniform1f(this.u('u_maskFeather'), p.mask.feather);
      gl.uniform1f(this.u('u_maskInvert'), p.mask.inverted ? 1 : 0);
    } else {
      gl.uniform1i(this.u('u_maskMode'), 0);
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    if (target) gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose(): void {
    const gl = this.gl;
    this.frameTex.forEach((t) => gl.deleteTexture(t));
    gl.deleteTexture(this.curvesTex);
    const luts = new Set([this.lutTex, ...this.lutSlots.map((x) => x.tex)]);
    luts.forEach((t) => gl.deleteTexture(t));
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
  }
}
