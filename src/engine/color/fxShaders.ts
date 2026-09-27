/**
 * Transition and effect passes (WebGL 2 / GLSL ES 3.00), FEATURES_V2 §6-8.
 * The exporter implements the same maths in src-tauri/src/render/transitions.rs and fx.rs.
 *
 * ============================================================================================
 *  CONVENTIONS AND THE CONSTANTS THE SPEC LEAVES OPEN (keep both sides identical)
 * ============================================================================================
 *  Space   uv in [0,1]^2 over the output frame. In the formulas below x runs left -> right and
 *          yT runs TOP -> bottom (image rows); the GLSL uses bottom-up uv and converts (yT = 1 - uv.y).
 *          W, H = output (project) size in pixels. "px" constants are OUTPUT pixels; the preview
 *          canvas is smaller and scales them (uv-space step = px / W).
 *  Colour  every mix happens on display-referred (sRGB-encoded) values in 0..1, like the rest of
 *          the pipeline: no linearisation. "linear in light" fades = linear ramps of these values.
 *  Outside Sampling outside [0,1]^2 (slides, pushes, zooms, shake, polaroid) returns BLACK.
 *          Blur taps clamp to the edge (texture CLAMP_TO_EDGE).
 *  Blur R  "R px of blur" everywhere = the house kernel of the clip blur (shaders.ts sampleFrame /
 *          sample.rs gaussian_blur): separable 7 taps at i * R/3 output px, i = -3..3, weights
 *          exp(-i^2 / 6), normalised; no blur when R < 0.5.
 *  smoothstep = GLSL Hermite: t = clamp((x-e0)/(e1-e0),0,1); t*t*(3-2t).
 *  LUMA    Rec.709 (0.2126, 0.7152, 0.0722).
 *  Noise   pcg(u32) = PCG-RXS-M-XS: s = v*747796405 + 2891336453; w = ((s >> ((s >> 28) + 4)) ^ s)
 *          * 277803737; pcg = (w >> 22) ^ w (all wrapping u32). rnd(i, seed) = pcg(i ^ pcg(seed)) /
 *          4294967295. valueNoise(x, seed) = 2 * mix(rnd(floor x), rnd(floor x + 1), f*f*(3-2f)) - 1.
 *          (TS: src/engine/effects.ts; pcg(0) = 129708002, pcg(1) = 2831084092.)
 *
 *  TRANSITIONS (p = easeInOut(raw), window centred on the cut, A outgoing, B incoming, both drawn
 *  as full frames over black with their own grade / transform / fades):
 *   dissolve      mix(A, B, p)
 *   dipToBlack    p < .5: mix(A, 0, 2p); else mix(0, B, 2p - 1)        (dipToWhite: through 1)
 *   wipe*         e = 0.02 (2 % soft edge along the wipe axis), q = p(1 + e) - e/2,
 *                 wB = 1 - smoothstep(q - e/2, q + e/2, c) with c = x (wipeRight), 1 - x (wipeLeft),
 *                 yT (wipeDown), 1 - yT (wipeUp); out = mix(A, B, wB)
 *   slideLeft     x >= 1-p: B(x - (1-p)) else A(x)          slideRight  x < p: B(x + 1 - p) else A(x)
 *   pushLeft      x < 1-p:  A(x + p) else B(x - (1-p))      pushRight   x < p: B(x + 1 - p) else A(x - p)
 *   zoomIn        A scaled 1 -> 1.3 (sA = 1 + .3p), B scaled .8 -> 1 (sB = .8 + .2p), about the
 *                 centre (sample at (uv - .5)/s + .5); out = mix(A', B', p)
 *   zoomOut       sA = 1 - .2p (1 -> .8), sB = 1.3 - .3p (1.3 -> 1); out = mix(A', B', p)
 *   blurDissolve  R = 12 (1 - |2p - 1|) px on both; out = mix(blur(A), blur(B), p)
 *   flash         w = (1 - |2p - 1|)^2; out = mix(mix(A, B, p), 1, w)
 *   circleOpen    d = |(uv - .5) * (W, H)| / (.5 |(W, H)|) (1 at the corners), e = 0.02,
 *                 R = p (1 + e), wB = 1 - smoothstep(R - e, R, d); out = mix(A, B, wB)
 *
 *  EFFECTS (s = intensity x envelope; the CPU part is effectFrame() in src/engine/effects.ts)
 *   envelope      120 ms linear ramp in and out: min(1, t/120ms, (D - t)/120ms), clamped >= 0.
 *                 Used by: blackAndWhite, sepia, letterbox, shake, rgbSplit, vhs, vignettePulse.
 *                 Own timing (s = intensity only): cameraSnap, fade*, zoomPunch, blurIn/Out, flashWhite.
 *   fadeFromBlack mix(c, 0, s (1 - u)), u = t / D     fadeToBlack  mix(c, 0, s u)   (White: to 1)
 *   flashWhite    mix(c, 1, s (1 - |2u - 1|))          (triangle, peak at D/2)
 *   blackAndWhite g = clamp((luma - .5) 1.1 + .5, 0, 1); mix(c, g, s)
 *   sepia         r' = .393r + .769g + .189b, g' = .349r + .686g + .168b, b' = .272r + .534g + .131b,
 *                 clamped; mix(c, sepia, s)
 *   letterbox     ratio k (default 2.39), frame aspect a = W/H. k > a: bars top and bottom, each
 *                 s (1 - a/k)/2 of the height; k < a: bars left and right, s (1 - k/a)/2 of the
 *                 width. Hard edges, black.
 *   shake         n_i = valueNoise(t * freq, i) for seeds 1, 2, 3; translation (n1, n2) * amp * s * W
 *                 px (right, down), rotation n3 * 100 * amp * s degrees (1 deg at the default amp,
 *                 clockwise on screen), overscan zoom 1 + 2 amp s; about the centre, in pixels
 *                 (aspect-correct): source = R(-rot)(P - T) / zoom, P = pixel - centre (y down).
 *   zoomPunch     scale 1 + .15 s k(u): k = easeOutBack(u / .35) for u < .35 (c1 = 1.70158),
 *                 else 1 - easeInOutCubic((u - .35) / .65); about the centre.
 *   blurIn / Out  R = 20 s (1 - u) / 20 s u px (house kernel).
 *   rgbSplit      o = amount s (1 + .5 valueNoise(8 t, 4)) (fraction of W; amount default .006);
 *                 R = src(x - o).r, G = src(x).g, B = src(x + o).b  (red image shifted right)
 *   vhs           row = floor(yT H), k = floor(30 t), bandY = fract(.25 t),
 *                 band = exp(-((yT - bandY) / .035)^2), jit = rnd(row, k) - .5,
 *                 dx = s (.0015 sin(2 pi (2 yT + 1.3 t)) + .02 band jit)   (fraction of W),
 *                 bleed = .002 s: R = src(x + dx + bleed).r, G = src(x + dx).g, B = src(x + dx - bleed).b,
 *                 scanlines c *= 1 - .2 s (.5 + .5 cos(2 pi yT H / 3)) (3 px period),
 *                 noise c += .08 s (rnd(px * 73856093 ^ py * 19349663, k) - .5) with (px, py) =
 *                 floor(x W), floor(yT H).
 *   vignettePulse v = s (.4 - .2 cos(2 pi t)) (0.2 <-> 0.6 at 1 Hz); the grade's vignette shape:
 *                 f = 1 - smoothstep(.35, 1.1, 1.35 |uv - .5|); c *= mix(1, f, v)
 *   cameraSnap    While a snap is active, the chain INPUT is the video composite at the snap's start
 *                 T0 (all video tracks, transitions and clip fades; before any effect) instead of
 *                 the composite at t; effects before the snap in track order apply to it at t.
 *                 The snap pass draws its input F as a polaroid:
 *                 flash alpha = s max(0, 1 - t / 250 ms); k = s easeOutCubic(min(1, t / 350 ms)).
 *                 photo scale sc = mix(1, scale, k) (scale default .92); border b = border k H px
 *                 (border default .03 = fraction of the HEIGHT); background = blur(F, 20 k px) *
 *                 (1 - .15 k); shadow: the card rect offset 0.012 H px downwards, alpha
 *                 .5 k (1 - smoothstep(0, .03 H, dist)) with dist the distance outside that rect;
 *                 c = bg (1 - shadow); inside the card (image + b on every side): white; inside the
 *                 image (sc W x sc H, centred): F((P / sc)); then mix(c, 1, flash alpha).
 *                 Shutter: procedural_shutter at T0, mixed at -6 dB (effects.ts proceduralShutter).
 *   Clip fades    Clip.fadeInMs / fadeOutMs: the layer colour is multiplied by
 *                 min(1, t / fadeIn) * min(1, (dur - t) / fadeOut) (each clamped to dur / 2, t
 *                 clamped to [0, dur]) before its opacity — the picture fades from / to black.
 * ============================================================================================
 */

export const FX_VERTEX_SHADER = `#version 300 es
precision highp float;
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const COMMON = `
uniform vec2 u_out;   // output (project) size in px
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
const float PI = 3.14159265358979;

bool inside(vec2 uv) { return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0; }

vec3 tex(sampler2D t, vec2 uv) { return inside(uv) ? texture(t, uv).rgb : vec3(0.0); }

// house blur kernel: 7x7 taps at i * R/3 output px, weights exp(-(i^2 + j^2) / 6)
vec3 blurTex(sampler2D t, vec2 uv, float r) {
  if (r < 0.5) return texture(t, uv).rgb;
  vec2 st = (r / 3.0) / u_out;
  vec3 acc = vec3(0.0);
  float total = 0.0;
  for (int i = -3; i <= 3; i++) {
    for (int j = -3; j <= 3; j++) {
      float w = exp(-float(i * i + j * j) / 6.0);
      acc += texture(t, uv + vec2(float(i), float(j)) * st).rgb * w;
      total += w;
    }
  }
  return acc / total;
}

uint pcg(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
float rnd(uint i, uint seed) { return float(pcg(i ^ pcg(seed))) / 4294967295.0; }
`;

export const TRANSITION_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform int u_type;
uniform float u_p;
${COMMON}

vec3 scaled(sampler2D t, vec2 uv, float s) { return tex(t, (uv - 0.5) / s + 0.5); }

float wipe(float c, float p) {
  float e = 0.02;
  float q = p * (1.0 + e) - e * 0.5;
  return 1.0 - smoothstep(q - e * 0.5, q + e * 0.5, c);
}

void main() {
  vec2 uv = v_uv;
  float p = u_p;
  float x = uv.x;
  float yT = 1.0 - uv.y;
  vec3 A = texture(u_a, uv).rgb;
  vec3 B = texture(u_b, uv).rgb;
  vec3 c;
  if (u_type == 0) c = mix(A, B, p);                                              // dissolve
  else if (u_type == 1) c = p < 0.5 ? mix(A, vec3(0.0), 2.0 * p) : mix(vec3(0.0), B, 2.0 * p - 1.0);   // dipToBlack
  else if (u_type == 2) c = p < 0.5 ? mix(A, vec3(1.0), 2.0 * p) : mix(vec3(1.0), B, 2.0 * p - 1.0);   // dipToWhite
  else if (u_type == 3) c = mix(A, B, wipe(1.0 - x, p));                        // wipeLeft
  else if (u_type == 4) c = mix(A, B, wipe(x, p));                              // wipeRight
  else if (u_type == 5) c = mix(A, B, wipe(1.0 - yT, p));                       // wipeUp
  else if (u_type == 6) c = mix(A, B, wipe(yT, p));                             // wipeDown
  else if (u_type == 7) c = x >= 1.0 - p ? tex(u_b, vec2(x - (1.0 - p), uv.y)) : A;          // slideLeft
  else if (u_type == 8) c = x < p ? tex(u_b, vec2(x + 1.0 - p, uv.y)) : A;                   // slideRight
  else if (u_type == 9) c = x < 1.0 - p ? tex(u_a, vec2(x + p, uv.y)) : tex(u_b, vec2(x - (1.0 - p), uv.y));   // pushLeft
  else if (u_type == 10) c = x < p ? tex(u_b, vec2(x + 1.0 - p, uv.y)) : tex(u_a, vec2(x - p, uv.y));        // pushRight
  else if (u_type == 11) c = mix(scaled(u_a, uv, 1.0 + 0.3 * p), scaled(u_b, uv, 0.8 + 0.2 * p), p);       // zoomIn
  else if (u_type == 12) c = mix(scaled(u_a, uv, 1.0 - 0.2 * p), scaled(u_b, uv, 1.3 - 0.3 * p), p);       // zoomOut
  else if (u_type == 13) {                                                        // blurDissolve
    float r = 12.0 * (1.0 - abs(2.0 * p - 1.0));
    c = mix(blurTex(u_a, uv, r), blurTex(u_b, uv, r), p);
  } else if (u_type == 14) {                                                      // flash
    float w = 1.0 - abs(2.0 * p - 1.0);
    c = mix(mix(A, B, p), vec3(1.0), w * w);
  } else {                                                                        // circleOpen
    float d = length((uv - 0.5) * u_out) / (0.5 * length(u_out));
    float e = 0.02;
    float R = p * (1.0 + e);
    c = mix(A, B, 1.0 - smoothstep(R - e, R, d));
  }
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

export const EFFECT_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform int u_type;
uniform float u_s;
uniform float u_t;
uniform float u_a;
uniform vec2 u_offset;
uniform float u_rot;
uniform float u_k;
uniform float u_scale;
uniform float u_border;
${COMMON}

// sample the source for a pixel offset P from the centre (y DOWN, output px)
vec3 atPixel(sampler2D t, vec2 P) {
  return tex(t, vec2(P.x / u_out.x + 0.5, 0.5 - P.y / u_out.y));
}

void main() {
  vec2 uv = v_uv;
  float s = u_s;
  vec3 c = texture(u_src, uv).rgb;
  float x = uv.x;
  float yT = 1.0 - uv.y;
  vec2 P = vec2(x - 0.5, yT - 0.5) * u_out; // pixel offset from the centre, y down

  if (u_type == 0) {                                                  // cameraSnap
    float k = u_k;
    float sc = mix(1.0, u_scale, k);
    float b = u_border * k * u_out.y;
    vec3 bg = blurTex(u_src, uv, 20.0 * k) * (1.0 - 0.15 * k);
    vec2 halfImg = 0.5 * sc * u_out;
    vec2 halfCard = halfImg + vec2(b);
    vec2 q = abs(P - vec2(0.0, 0.012 * u_out.y)) - halfCard;
    float dist = length(max(q, vec2(0.0)));
    float shadow = 0.5 * k * (1.0 - smoothstep(0.0, 0.03 * u_out.y, dist));
    c = bg * (1.0 - shadow);
    if (abs(P.x) <= halfCard.x && abs(P.y) <= halfCard.y) c = vec3(1.0);
    if (abs(P.x) <= halfImg.x && abs(P.y) <= halfImg.y) c = atPixel(u_src, P / sc);
    c = mix(c, vec3(1.0), u_a);
  } else if (u_type == 1 || u_type == 2) {                            // fadeFromBlack / fadeToBlack
    c = mix(c, vec3(0.0), u_a);
  } else if (u_type == 3 || u_type == 4 || u_type == 15) {            // fadeFromWhite / fadeToWhite / flashWhite
    c = mix(c, vec3(1.0), u_a);
  } else if (u_type == 5) {                                           // blackAndWhite
    float g = clamp((dot(c, LUMA) - 0.5) * 1.1 + 0.5, 0.0, 1.0);
    c = mix(c, vec3(g), s);
  } else if (u_type == 6) {                                           // sepia
    vec3 sp = vec3(dot(c, vec3(0.393, 0.769, 0.189)), dot(c, vec3(0.349, 0.686, 0.168)), dot(c, vec3(0.272, 0.534, 0.131)));
    c = mix(c, clamp(sp, 0.0, 1.0), s);
  } else if (u_type == 7) {                                           // letterbox
    float a = u_out.x / u_out.y;
    float k = u_k;
    if (k > a) {
      float bar = s * (1.0 - a / k) * 0.5;
      if (yT < bar || yT > 1.0 - bar) c = vec3(0.0);
    } else if (k < a) {
      float bar = s * (1.0 - k / a) * 0.5;
      if (x < bar || x > 1.0 - bar) c = vec3(0.0);
    }
  } else if (u_type == 8) {                                           // shake
    float th = radians(u_rot);
    vec2 d = P - u_offset * u_out.x;
    // inverse of a clockwise-on-screen rotation (y down): R(-th)
    vec2 r = vec2(cos(th) * d.x + sin(th) * d.y, -sin(th) * d.x + cos(th) * d.y);
    c = atPixel(u_src, r / u_a);
  } else if (u_type == 9) {                                           // zoomPunch
    c = atPixel(u_src, P / u_a);
  } else if (u_type == 10 || u_type == 11) {                          // blurIn / blurOut
    c = blurTex(u_src, uv, u_a);
  } else if (u_type == 12) {                                          // rgbSplit
    float o = u_offset.x;
    c = vec3(tex(u_src, vec2(x - o, uv.y)).r, c.g, tex(u_src, vec2(x + o, uv.y)).b);
  } else if (u_type == 13) {                                          // vhs
    float row = floor(yT * u_out.y);
    uint k = uint(floor(u_t * 30.0));
    float bandY = fract(0.25 * u_t);
    float z = (yT - bandY) / 0.035;
    float band = exp(-z * z);
    float jit = rnd(uint(max(row, 0.0)), k) - 0.5;
    float dx = s * (0.0015 * sin(2.0 * PI * (2.0 * yT + 1.3 * u_t)) + 0.02 * band * jit);
    float bleed = 0.002 * s;
    c = vec3(tex(u_src, vec2(x + dx + bleed, uv.y)).r, tex(u_src, vec2(x + dx, uv.y)).g, tex(u_src, vec2(x + dx - bleed, uv.y)).b);
    c *= 1.0 - 0.2 * s * (0.5 + 0.5 * cos(2.0 * PI * yT * u_out.y / 3.0));
    uvec2 px = uvec2(max(floor(vec2(x * u_out.x, yT * u_out.y)), vec2(0.0)));
    c += 0.08 * s * (rnd((px.x * 73856093u) ^ (px.y * 19349663u), k) - 0.5);
  } else if (u_type == 14) {                                          // vignettePulse
    float f = 1.0 - smoothstep(0.35, 1.1, length(uv - 0.5) * 1.35);
    c *= mix(1.0, f, u_a);
  }
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

export const COPY_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
void main() {
  outColor = vec4(texture(u_src, v_uv).rgb, 1.0);
}`;
