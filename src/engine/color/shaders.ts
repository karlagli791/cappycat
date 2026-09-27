/**
 * GPU color pipeline (WebGL 2 / GLSL ES 3.00).
 *
 * Stage order follows the design spec:
 *   Primary adjustments -> Temperature & white balance -> 3-way wheels (lift/gamma/gain/offset)
 *   -> HSL 8-color tuning -> RGB curves -> 3D LUT -> Grain & vignette -> output
 */

export const VERTEX_SHADER = `#version 300 es
precision highp float;
in vec2 a_pos;        // clip-space quad (-1..1)
uniform mat3 u_uvTransform; // maps quad uv (0..1) -> source uv (0..1), includes crop+transform
out vec2 v_uv;
out vec2 v_canvasUv;
void main() {
  vec2 uv = a_pos * 0.5 + 0.5;
  vec3 s = u_uvTransform * vec3(uv, 1.0);
  v_uv = s.xy;
  v_canvasUv = uv;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

export const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 v_uv;
in vec2 v_canvasUv;
out vec4 outColor;

uniform sampler2D u_frame;
uniform sampler2D u_curves;   // 256 x 4 : master, r, g, b
uniform sampler3D u_lut;
uniform float u_useLut;
uniform float u_lutSize;
uniform float u_lutIntensity;
uniform vec2 u_texel;

uniform float u_exposure;    // stops, -3..3
uniform float u_contrast;    // -1..1
uniform float u_brightness;  // -1..1
uniform float u_highlights;  // -1..1
uniform float u_shadows;     // -1..1
uniform float u_brilliance;  // -1..1
uniform float u_saturation;  // -1..1
uniform float u_vibrance;    // -1..1
uniform float u_sharpness;   // 0..1
uniform float u_temperature; // -1..1
uniform float u_tint;        // -1..1
uniform vec3 u_lift;
uniform vec3 u_gamma;
uniform vec3 u_gain;
uniform vec3 u_offset;
uniform vec3 u_hsl[8];       // h (-1..1 => -180..180deg), s (-1..1), l (-1..1)
uniform float u_vignette;    // 0..1
uniform float u_grain;       // 0..1
uniform float u_time;
uniform float u_opacity;
uniform float u_fade;        // clip video fade: 1 = none, 0 = black (FEATURES_V2 §8)
uniform float u_blur;        // px

uniform int u_maskMode;      // 0 none, 1 rect, 2 circle, 3 split, 4 filmstrip
uniform vec4 u_maskRect;     // x,y,w,h normalised
uniform float u_maskFeather;
uniform float u_maskInvert;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 sampleFrame(vec2 uv) {
  if (u_blur > 0.5) {
    vec3 acc = vec3(0.0);
    float total = 0.0;
    float r = min(u_blur, 12.0);
    for (int i = -3; i <= 3; i++) {
      for (int j = -3; j <= 3; j++) {
        vec2 o = vec2(float(i), float(j)) * u_texel * r / 3.0;
        float w = exp(-float(i * i + j * j) / 6.0);
        acc += texture(u_frame, uv + o).rgb * w;
        total += w;
      }
    }
    return acc / total;
  }
  return texture(u_frame, uv).rgb;
}

float maskWeight(vec2 uv) {
  if (u_maskMode == 0) return 1.0;
  float f = max(u_maskFeather, 0.0005);
  float w = 1.0;
  vec2 c = u_maskRect.xy + u_maskRect.zw * 0.5;
  if (u_maskMode == 1) {
    vec2 d = abs(uv - c) - u_maskRect.zw * 0.5;
    float m = max(d.x, d.y);
    w = 1.0 - smoothstep(0.0, f, m);
  } else if (u_maskMode == 2) {
    vec2 d = (uv - c) / max(u_maskRect.zw * 0.5, vec2(1e-4));
    float m = length(d) - 1.0;
    w = 1.0 - smoothstep(0.0, f * 2.0, m);
  } else if (u_maskMode == 3) {
    w = 1.0 - smoothstep(0.0, f, uv.x - u_maskRect.x);
  } else if (u_maskMode == 4) {
    float dy = abs(uv.y - c.y) - u_maskRect.w * 0.5;
    w = 1.0 - smoothstep(0.0, f, dy);
  }
  return mix(w, 1.0 - w, u_maskInvert);
}

void main() {
  vec2 uv = v_uv;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    outColor = vec4(0.0);
    return;
  }
  vec3 c = sampleFrame(uv);

  // ---- Primary adjustments ----
  c *= exp2(u_exposure);
  // highlight roll-off when brightening: values above the knee ease towards white instead of
  // clipping (mirrors exposure_rolloff() in src-tauri/src/render/color.rs)
  if (u_exposure > 0.0) {
    vec3 over = max(c - 0.8, 0.0);
    c = min(c, vec3(0.8)) + 0.2 * tanh(over / 0.2);
  }
  c += u_brightness * 0.5;
  c = (c - 0.5) * (1.0 + u_contrast) + 0.5;
  float luma = dot(c, LUMA);
  float hiW = smoothstep(0.5, 1.0, luma);
  float shW = 1.0 - smoothstep(0.0, 0.5, luma);
  c += u_highlights * 0.5 * hiW;
  c += u_shadows * 0.5 * shW;

  // ---- Brilliance (mirrors brilliance() in src-tauri/src/render/color.rs) ----
  if (u_brilliance != 0.0) {
    float bl = clamp(dot(c, LUMA), 0.0, 1.0);
    float lift = 0.6 * bl * (1.0 - bl) * (1.0 - bl);
    float comp = 0.4 * bl * bl * (1.0 - bl);
    float nl = bl + u_brilliance * (lift - comp);
    if (bl > 1e-4) c *= nl / bl; else c += vec3(nl);
    float bl2 = dot(c, LUMA);
    c = mix(vec3(bl2), c, 1.0 + 0.15 * u_brilliance);
  }

  if (u_sharpness > 0.0) {
    vec3 blur = (texture(u_frame, uv + vec2(u_texel.x, 0.0)).rgb +
                 texture(u_frame, uv - vec2(u_texel.x, 0.0)).rgb +
                 texture(u_frame, uv + vec2(0.0, u_texel.y)).rgb +
                 texture(u_frame, uv - vec2(0.0, u_texel.y)).rgb) * 0.25;
    c += (texture(u_frame, uv).rgb - blur) * u_sharpness * 2.0;
  }

  // ---- Temperature & tint ----
  c.r += u_temperature * 0.12;
  c.b -= u_temperature * 0.12;
  c.g += u_tint * -0.1;
  c.r += u_tint * 0.05;
  c.b += u_tint * 0.05;

  // ---- 3-way color wheels (ASC CDL-like lift/gamma/gain + offset) ----
  c = c * (1.0 + u_gain) + u_lift * (1.0 - c) + u_offset;
  c = max(c, vec3(0.0));
  c = pow(c, vec3(1.0) / max(vec3(1.0) + u_gamma, vec3(0.05)));

  // ---- Saturation & vibrance ----
  luma = dot(c, LUMA);
  float sat = length(c - vec3(luma));
  float vib = u_vibrance * (1.0 - clamp(sat * 1.5, 0.0, 1.0));
  c = mix(vec3(luma), c, 1.0 + u_saturation + vib);

  // ---- HSL 8-color tuning ----
  vec3 hsv = rgb2hsv(clamp(c, 0.0, 1.0));
  float hueDeg = hsv.x * 360.0;
  float centers[8];
  centers[0] = 0.0; centers[1] = 30.0; centers[2] = 60.0; centers[3] = 120.0;
  centers[4] = 180.0; centers[5] = 240.0; centers[6] = 270.0; centers[7] = 300.0;
  float dh = 0.0; float ds = 0.0; float dl = 0.0;
  for (int i = 0; i < 8; i++) {
    float d = abs(hueDeg - centers[i]);
    d = min(d, 360.0 - d);
    float w = 1.0 - smoothstep(0.0, 35.0, d);
    w *= smoothstep(0.0, 0.25, hsv.y); // grey pixels are not affected
    dh += u_hsl[i].x * w;
    ds += u_hsl[i].y * w;
    dl += u_hsl[i].z * w;
  }
  hsv.x = fract(hsv.x + dh * (30.0 / 360.0)); // CapCut scale: +-100 = +-30 deg
  hsv.y = clamp(hsv.y * (1.0 + ds), 0.0, 1.0);
  hsv.z = clamp(hsv.z * (1.0 + dl * 0.5), 0.0, 1.0);
  c = hsv2rgb(hsv);

  // ---- RGB curves ----
  c = clamp(c, 0.0, 1.0);
  c.r = texture(u_curves, vec2(c.r, 0.375)).r;
  c.g = texture(u_curves, vec2(c.g, 0.625)).r;
  c.b = texture(u_curves, vec2(c.b, 0.875)).r;
  c.r = texture(u_curves, vec2(c.r, 0.125)).r;
  c.g = texture(u_curves, vec2(c.g, 0.125)).r;
  c.b = texture(u_curves, vec2(c.b, 0.125)).r;

  // ---- 3D LUT ----
  if (u_useLut > 0.5) {
    float s = u_lutSize;
    vec3 lc = c * ((s - 1.0) / s) + 0.5 / s;
    vec3 graded = texture(u_lut, lc).rgb;
    c = mix(c, graded, u_lutIntensity);
  }

  // ---- Grain & vignette (in output-frame space) ----
  if (u_vignette > 0.0) {
    vec2 d = v_canvasUv - 0.5;
    float v = 1.0 - smoothstep(0.35, 1.1, length(d) * 1.35);
    c *= mix(1.0, v, u_vignette);
  }
  if (u_grain > 0.0) {
    float n = hash(v_canvasUv * 1024.0 + fract(u_time * 0.001) * 100.0) - 0.5;
    c += n * u_grain * 0.25;
  }

  // clip fade from / to black (before the opacity: the picture itself goes black)
  c *= u_fade;

  float alpha = u_opacity * maskWeight(v_canvasUv);
  outColor = vec4(clamp(c, 0.0, 1.0) * alpha, alpha);
}`;
