/**
 * Off-screen passes on top of the colour renderer: transitions (two layers -> one frame) and the
 * effect chain (frame -> frame, ping-pong), plus the frozen frame of a camera snap. Shares the
 * WebGL2 context of the ColorRenderer. Only used while a transition / effect is on screen: the
 * plain path draws the layer straight to the canvas.
 */
import type { EffectFrame } from '../effects';
import type { RenderTarget } from './renderer';
import { COPY_FRAGMENT_SHADER, EFFECT_FRAGMENT_SHADER, FX_VERTEX_SHADER, TRANSITION_FRAGMENT_SHADER } from './fxShaders';

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

class Pass {
  readonly program: WebGLProgram;
  private uniforms = new Map<string, WebGLUniformLocation | null>();
  readonly vao: WebGLVertexArrayObject;
  constructor(private gl: WebGL2RenderingContext, fragment: string) {
    const program = gl.createProgram();
    if (!program) throw new Error('createProgram failed');
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, FX_VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`);
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
  }
  u(name: string): WebGLUniformLocation | null {
    if (!this.uniforms.has(name)) this.uniforms.set(name, this.gl.getUniformLocation(this.program, name));
    return this.uniforms.get(name) ?? null;
  }
  dispose() {
    this.gl.deleteProgram(this.program);
    this.gl.deleteVertexArray(this.vao);
  }
}

export type TargetName = 'a' | 'b' | 'x' | 'y' | 'snap';

export class FxCompositor {
  private transitionPass: Pass;
  private effectPass: Pass;
  private copyPass: Pass;
  private targets = new Map<TargetName, RenderTarget>();
  private w = 0;
  private h = 0;

  constructor(private gl: WebGL2RenderingContext) {
    this.transitionPass = new Pass(gl, TRANSITION_FRAGMENT_SHADER);
    this.effectPass = new Pass(gl, EFFECT_FRAGMENT_SHADER);
    this.copyPass = new Pass(gl, COPY_FRAGMENT_SHADER);
  }

  /** Canvas-sized targets (re-created when the canvas size changes). */
  target(name: TargetName, w: number, h: number): RenderTarget {
    if (w !== this.w || h !== this.h) {
      this.targets.forEach((t) => this.free(t));
      this.targets.clear();
      this.w = w;
      this.h = h;
    }
    let t = this.targets.get(name);
    if (!t) {
      t = this.create(w, h);
      this.targets.set(name, t);
    }
    return t;
  }

  private create(w: number, h: number): RenderTarget {
    const gl = this.gl;
    const tex = gl.createTexture();
    const fbo = gl.createFramebuffer();
    if (!tex || !fbo) throw new Error('createTexture / createFramebuffer failed');
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex, w, h };
  }

  private free(t: RenderTarget) {
    this.gl.deleteTexture(t.tex);
    this.gl.deleteFramebuffer(t.fbo);
  }

  private begin(pass: Pass, out: RenderTarget | null, canvasW: number, canvasH: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, out?.fbo ?? null);
    gl.viewport(0, 0, out?.w ?? canvasW, out?.h ?? canvasH);
    gl.disable(gl.BLEND);
    gl.useProgram(pass.program);
    gl.bindVertexArray(pass.vao);
  }

  private end() {
    const gl = this.gl;
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.BLEND);
  }

  private bind(unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | null) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(loc, unit);
  }

  /** A -> B transition of type `code` (TRANSITION_TYPES order) at eased progress p. */
  transition(code: number, p: number, a: RenderTarget, b: RenderTarget, out: RenderTarget | null, outSize: [number, number], canvasW: number, canvasH: number) {
    const pass = this.transitionPass;
    this.begin(pass, out, canvasW, canvasH);
    const gl = this.gl;
    this.bind(4, a.tex, pass.u('u_a'));
    this.bind(5, b.tex, pass.u('u_b'));
    gl.uniform1i(pass.u('u_type'), code);
    gl.uniform1f(pass.u('u_p'), p);
    gl.uniform2f(pass.u('u_out'), outSize[0], outSize[1]);
    this.end();
  }

  /** One effect pass (EFFECT_INFO order `code`) from `src` into `out` (null: the canvas). */
  effect(code: number, f: EffectFrame, src: RenderTarget, out: RenderTarget | null, outSize: [number, number], canvasW: number, canvasH: number) {
    const pass = this.effectPass;
    this.begin(pass, out, canvasW, canvasH);
    const gl = this.gl;
    this.bind(4, src.tex, pass.u('u_src'));
    gl.uniform1i(pass.u('u_type'), code);
    gl.uniform1f(pass.u('u_s'), f.s);
    gl.uniform1f(pass.u('u_t'), f.t);
    gl.uniform1f(pass.u('u_a'), f.a);
    gl.uniform2f(pass.u('u_offset'), f.offset[0], f.offset[1]);
    gl.uniform1f(pass.u('u_rot'), f.rot);
    gl.uniform1f(pass.u('u_k'), f.k);
    gl.uniform1f(pass.u('u_scale'), f.scale);
    gl.uniform1f(pass.u('u_border'), f.border);
    gl.uniform2f(pass.u('u_out'), outSize[0], outSize[1]);
    this.end();
  }

  /** Copy a target (to another target or the canvas). */
  copy(src: RenderTarget, out: RenderTarget | null, canvasW: number, canvasH: number) {
    const pass = this.copyPass;
    this.begin(pass, out, canvasW, canvasH);
    this.bind(4, src.tex, pass.u('u_src'));
    this.end();
  }

  dispose() {
    this.targets.forEach((t) => this.free(t));
    this.targets.clear();
    this.transitionPass.dispose();
    this.effectPass.dispose();
    this.copyPass.dispose();
  }
}
