import { useEffect, useRef, useState } from 'react';
import { clipDurationMs, useEditor, useSelectedClip, type KeyframeProperty } from '@/state/store';
import { evaluate, sampleEasing, type BezierHandles } from '@/engine/keyframes';
import { SPEED_PRESETS, outputDuration, presetCurve, speedAt } from '@/engine/speed';
import type { Easing, Keyframe, Keyframed, SpeedPoint, SpeedPreset } from '@/types/project';
import { THEME } from '@/lib/theme';
import { CloseIcon } from './Overlays';

const PROPS: Array<{ key: KeyframeProperty; label: string; min: number; max: number; comp?: number }> = [
  { key: 'position', label: 'Position X', min: -1, max: 1, comp: 0 },
  { key: 'position', label: 'Position Y', min: -1, max: 1, comp: 1 },
  { key: 'scale', label: 'Scale', min: 0, max: 3 },
  { key: 'rotation', label: 'Rotation', min: -180, max: 180 },
  { key: 'opacity', label: 'Opacity', min: 0, max: 1 },
  { key: 'blur', label: 'Blur', min: 0, max: 40 },
];

const EASINGS: Easing[] = ['linear', 'easeIn', 'easeOut', 'easeInOut', 'bounce', 'elastic', 'bezier'];

type Kf = Keyframed<number | number[]>;

export default function KeyframeDrawer({ height }: { height: number }) {
  const open = useEditor((s) => s.drawerOpen);
  const mode = useEditor((s) => s.drawerMode);
  const setDrawer = useEditor((s) => s.setDrawer);
  const label = useEditor((s) => {
    const id = s.selection.clipIds[0];
    if (!id) return null;
    for (const t of s.project.tracks) for (const c of t.clips) if (c.id === id) return c.label ?? 'clip';
    return null;
  });

  return (
    <div className={`drawer ${open ? 'open' : ''}`} style={{ height: open ? height : 0 }}>
      <div className="bar">
        <button className={`small ${mode === 'keyframes' ? 'active' : ''}`} onClick={() => setDrawer(true, 'keyframes')}>
          Keyframe graph
        </button>
        <button className={`small ${mode === 'speed' ? 'active' : ''}`} onClick={() => setDrawer(true, 'speed')}>
          Speed curve
        </button>
        <span className="spacer" />
        {label ? <span>{label}</span> : <span>Select a clip</span>}
        <span className="kbd">Alt+K</span>
        <button className="small ghost" onClick={() => setDrawer(false)} title="Close (Esc)" aria-label="Close the graph">
          <CloseIcon />
        </button>
      </div>
      {open ? mode === 'keyframes' ? <KeyframeGraph /> : <SpeedGraph /> : <div />}
    </div>
  );
}

function useCanvasSize(wrapRef: React.RefObject<HTMLDivElement | null>) {
  const [size, setSize] = useState({ w: 600, h: 180 });
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize((s) => {
        const w = Math.max(100, Math.floor(r.width));
        const h = Math.max(60, Math.floor(r.height));
        return s.w === w && s.h === h ? s : { w, h };
      });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [wrapRef]);
  return size;
}

function prepCanvas(canvas: HTMLCanvasElement, size: { w: number; h: number }): CanvasRenderingContext2D {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.round(size.w * dpr);
  const H = Math.round(size.h * dpr);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/* ------------------------------ keyframes ------------------------------ */

function KeyframeGraph() {
  const sel = useSelectedClip();
  const playheadMs = useEditor((s) => s.playheadMs);
  const drawerProperty = useEditor((s) => s.drawerProperty);
  const [propIdx, setPropIdx] = useState(() => Math.max(0, PROPS.findIndex((p) => p.key === drawerProperty)));
  const [selKf, setSelKf] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const size = useCanvasSize(wrapRef);
  const dragRef = useRef<{ type: 'kf' | 'h1' | 'h2'; index: number; orig: Keyframe<number | number[]>[] } | null>(null);

  useEffect(() => {
    const i = PROPS.findIndex((p) => p.key === drawerProperty);
    if (i >= 0 && PROPS[i].key !== PROPS[propIdx].key) setPropIdx(i);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawerProperty]);

  const prop = PROPS[propIdx];
  const clip = sel?.clip;
  const kf = clip ? (clip.transform[prop.key] as Kf) : null;
  const dur = clip ? clipDurationMs(clip) : 1;
  const PAD = { l: 44, r: 12, t: 12, b: 18 };

  const valOf = (v: number | number[]): number => (Array.isArray(v) ? v[prop.comp ?? 0] : v);
  const tx = (ms: number) => PAD.l + (ms / dur) * (size.w - PAD.l - PAD.r);
  const ty = (v: number) => PAD.t + (1 - (v - prop.min) / (prop.max - prop.min)) * (size.h - PAD.t - PAD.b);
  const xt = (x: number) => ((x - PAD.l) / (size.w - PAD.l - PAD.r)) * dur;
  const yv = (y: number) => prop.min + (1 - (y - PAD.t) / (size.h - PAD.t - PAD.b)) * (prop.max - prop.min);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = prepCanvas(canvas, size);
    ctx.fillStyle = THEME.bg1;
    ctx.fillRect(0, 0, size.w, size.h);
    ctx.strokeStyle = THEME.line;
    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.fillStyle = THEME.textFaint;
    for (let i = 0; i <= 4; i++) {
      const v = prop.min + ((prop.max - prop.min) * i) / 4;
      const y = Math.round(ty(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD.l, y);
      ctx.lineTo(size.w - PAD.r, y);
      ctx.stroke();
      ctx.fillText(fmt(v), 4, y + 3);
    }
    if (!clip || !kf) {
      ctx.fillStyle = THEME.textFaint;
      ctx.fillText('Select a clip to edit keyframes', PAD.l + 8, size.h / 2);
      return;
    }
    ctx.strokeStyle = THEME.curve;
    ctx.lineWidth = 2;
    ctx.beginPath();
    const N = 200;
    for (let i = 0; i <= N; i++) {
      const ms = (i / N) * dur;
      const v = valOf(evaluate(kf, ms));
      if (i === 0) ctx.moveTo(tx(ms), ty(v));
      else ctx.lineTo(tx(ms), ty(v));
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    kf.keyframes.forEach((k, i) => {
      const x = tx(k.timeMs);
      const y = ty(valOf(k.value));
      ctx.fillStyle = selKf === i ? THEME.keyframeSelected : THEME.keyframe;
      ctx.beginPath();
      ctx.moveTo(x, y - 6);
      ctx.lineTo(x + 6, y);
      ctx.lineTo(x, y + 6);
      ctx.lineTo(x - 6, y);
      ctx.closePath();
      ctx.fill();
      if (selKf === i && i < kf.keyframes.length - 1 && k.easing === 'bezier') {
        const next = kf.keyframes[i + 1];
        const h = k.bezier ?? [0.25, 0.1, 0.25, 1];
        const x0 = tx(k.timeMs);
        const y0 = ty(valOf(k.value));
        const x1 = tx(next.timeMs);
        const y1 = ty(valOf(next.value));
        const hx1 = x0 + (x1 - x0) * h[0];
        const hy1 = y0 + (y1 - y0) * h[1];
        const hx2 = x0 + (x1 - x0) * h[2];
        const hy2 = y0 + (y1 - y0) * h[3];
        ctx.strokeStyle = THEME.keyframeSelected;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(hx1, hy1);
        ctx.moveTo(x1, y1);
        ctx.lineTo(hx2, hy2);
        ctx.stroke();
        ctx.fillStyle = THEME.keyframeSelected;
        for (const [hx, hy] of [[hx1, hy1], [hx2, hy2]]) {
          ctx.beginPath();
          ctx.arc(hx, hy, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });
    const localMs = playheadMs - clip.startMs;
    if (localMs >= 0 && localMs <= dur) {
      const x = Math.round(tx(localMs)) + 0.5;
      ctx.strokeStyle = THEME.playhead;
      ctx.beginPath();
      ctx.moveTo(x, PAD.t);
      ctx.lineTo(x, size.h - PAD.b);
      ctx.stroke();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clip, kf, size, prop, playheadMs, selKf]);

  const hitKeyframe = (x: number, y: number): number | null => {
    if (!kf) return null;
    for (let i = 0; i < kf.keyframes.length; i++) {
      const k = kf.keyframes[i];
      if (Math.abs(tx(k.timeMs) - x) < 8 && Math.abs(ty(valOf(k.value)) - y) < 8) return i;
    }
    return null;
  };

  const hitHandle = (x: number, y: number): 'h1' | 'h2' | null => {
    if (!kf || selKf == null || selKf >= kf.keyframes.length - 1) return null;
    const k = kf.keyframes[selKf];
    if (k.easing !== 'bezier') return null;
    const next = kf.keyframes[selKf + 1];
    const h = k.bezier ?? [0.25, 0.1, 0.25, 1];
    const x0 = tx(k.timeMs);
    const y0 = ty(valOf(k.value));
    const x1 = tx(next.timeMs);
    const y1 = ty(valOf(next.value));
    const pts: Array<['h1' | 'h2', number, number]> = [
      ['h1', x0 + (x1 - x0) * h[0], y0 + (y1 - y0) * h[1]],
      ['h2', x0 + (x1 - x0) * h[2], y0 + (y1 - y0) * h[3]],
    ];
    for (const [id, hx, hy] of pts) if (Math.hypot(hx - x, hy - y) < 7) return id;
    return null;
  };

  const withValue = (k: Keyframe<number | number[]>, v: number): number | number[] => {
    if (Array.isArray(k.value)) {
      const arr = [...k.value];
      arr[prop.comp ?? 0] = v;
      return arr;
    }
    return v;
  };

  /** Replace the property's keyframes (inside the running gesture). */
  const writeKeys = (keys: Keyframe<number | number[]>[]) => {
    if (!clip) return;
    const sorted = [...keys].sort((a, b) => a.timeMs - b.timeMs);
    useEditor.getState().updateClip(clip.id, (c) => ({ transform: { ...c.transform, [prop.key]: { ...(c.transform[prop.key] as Kf), keyframes: sorted } } }));
    return sorted;
  };

  const pos = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!clip || !kf) return;
    const { x, y } = pos(e);
    const st = useEditor.getState();
    const handle = hitHandle(x, y);
    if (handle) {
      canvasRef.current!.setPointerCapture(e.pointerId);
      st.beginGesture();
      dragRef.current = { type: handle, index: selKf!, orig: kf.keyframes };
      return;
    }
    const i = hitKeyframe(x, y);
    if (i != null) {
      if (e.button === 2 || e.altKey) {
        st.removeTransformKeyframe(clip.id, prop.key, kf.keyframes[i].timeMs);
        setSelKf(null);
        return;
      }
      canvasRef.current!.setPointerCapture(e.pointerId);
      st.beginGesture();
      setSelKf(i);
      dragRef.current = { type: 'kf', index: i, orig: kf.keyframes };
      return;
    }
    if (x > PAD.l && e.button === 0) {
      // add a keyframe at the clicked time/value, and keep dragging it
      const ms = Math.max(0, Math.min(dur, xt(x)));
      const v = Math.max(prop.min, Math.min(prop.max, yv(y)));
      const cur = evaluate(kf, ms);
      canvasRef.current!.setPointerCapture(e.pointerId);
      st.beginGesture();
      const added: Keyframe<number | number[]> = { timeMs: ms, value: withValue({ timeMs: ms, value: cur, easing: 'easeInOut' }, v), easing: 'easeInOut' };
      const keys = writeKeys([...kf.keyframes.filter((k) => Math.abs(k.timeMs - ms) > 0.5), added])!;
      const idx = keys.indexOf(added);
      setSelKf(idx);
      dragRef.current = { type: 'kf', index: idx, orig: keys };
      st.setPlayhead(clip.startMs + ms);
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d || !clip) return;
    const { x, y } = pos(e);
    const k = d.orig[d.index];
    if (!k) return;
    if (d.type === 'kf') {
      // stay between the neighbours so the order (and the selection) never changes
      const lo = d.index > 0 ? d.orig[d.index - 1].timeMs + 1 : 0;
      const hi = d.index < d.orig.length - 1 ? d.orig[d.index + 1].timeMs - 1 : dur;
      const ms = Math.max(lo, Math.min(hi, xt(x)));
      const v = Math.max(prop.min, Math.min(prop.max, yv(y)));
      writeKeys(d.orig.map((o, j) => (j === d.index ? { ...o, timeMs: ms, value: withValue(o, v) } : o)));
    } else if (d.index < d.orig.length - 1) {
      const next = d.orig[d.index + 1];
      const x0 = tx(k.timeMs);
      const y0 = ty(valOf(k.value));
      const x1 = tx(next.timeMs);
      const y1 = ty(valOf(next.value));
      const hx = Math.max(0, Math.min(1, (x - x0) / (x1 - x0 || 1)));
      const hy = y1 === y0 ? 0.5 : (y - y0) / (y1 - y0);
      const h: BezierHandles = [...(k.bezier ?? [0.25, 0.1, 0.25, 1])] as BezierHandles;
      if (d.type === 'h1') {
        h[0] = hx;
        h[1] = hy;
      } else {
        h[2] = hx;
        h[3] = hy;
      }
      writeKeys(d.orig.map((o, j) => (j === d.index ? { ...o, easing: 'bezier' as Easing, bezier: h } : o)));
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    if (canvasRef.current?.hasPointerCapture(e.pointerId)) canvasRef.current.releasePointerCapture(e.pointerId);
    useEditor.getState().endGesture();
  };

  const selected = kf && selKf != null ? kf.keyframes[selKf] : null;

  return (
    <div className="drawer-body" style={{ gridTemplateColumns: '160px 1fr' }}>
      <div className="drawer-side">
        <select
          value={propIdx}
          onChange={(e) => {
            setPropIdx(Number(e.target.value));
            setSelKf(null);
            useEditor.getState().setDrawer(true, 'keyframes', PROPS[Number(e.target.value)].key);
          }}
          style={{ width: '100%', marginBottom: 6 }}
        >
          {PROPS.map((p, i) => (
            <option key={p.label} value={i}>
              {p.label}
            </option>
          ))}
        </select>
        {selected && clip ? (
          <>
            <div className="hint" style={{ marginBottom: 4 }}>
              Keyframe @ {(selected.timeMs / 1000).toFixed(2)}s
            </div>
            <select
              value={selected.easing}
              onChange={(e) => useEditor.getState().setTransformKeyframe(clip.id, prop.key, selected.timeMs, selected.value, e.target.value as Easing, selected.bezier)}
              style={{ width: '100%' }}
            >
              {EASINGS.map((ez) => (
                <option key={ez} value={ez}>
                  {ez}
                </option>
              ))}
            </select>
            <div className="hint" style={{ marginTop: 6 }}>
              {selected.easing === 'bezier' ? 'Drag the blue handles to shape the outgoing tangent.' : 'Easing shapes the segment after this keyframe. Choose "bezier" for custom handles.'}
            </div>
            <button
              className="small"
              style={{ marginTop: 6, width: '100%' }}
              onClick={() => {
                useEditor.getState().removeTransformKeyframe(clip.id, prop.key, selected.timeMs);
                setSelKf(null);
              }}
            >
              Remove keyframe
            </button>
          </>
        ) : (
          <div className="hint">Click the graph to add a keyframe. Drag to move. Alt-click to remove.</div>
        )}
        <EasingPreview easing={selected?.easing ?? 'easeInOut'} handles={selected?.bezier} />
      </div>
      <div ref={wrapRef} style={{ minHeight: 0, minWidth: 0 }}>
        <canvas ref={canvasRef} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onContextMenu={(e) => e.preventDefault()} />
      </div>
    </div>
  );
}

function EasingPreview({ easing, handles }: { easing: Easing; handles?: BezierHandles }) {
  const pts = sampleEasing(easing, handles, 48);
  const d = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${(x * 60).toFixed(1)},${(40 - y * 36 + 2).toFixed(1)}`).join(' ');
  return (
    <svg width="60" height="44" style={{ marginTop: 8, background: 'var(--bg-2)', borderRadius: 4 }}>
      <path d={d} fill="none" stroke={THEME.curve} strokeWidth="1.5" />
    </svg>
  );
}

function fmt(v: number): string {
  return Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);
}

/* -------------------------------- speed -------------------------------- */

function SpeedGraph() {
  const sel = useSelectedClip();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const size = useCanvasSize(wrapRef);
  const dragRef = useRef<{ index: number; orig: SpeedPoint[] } | null>(null);
  const PAD = { l: 44, r: 12, t: 10, b: 16 };

  const clip = sel?.clip;
  const curve = clip?.speed;
  const lg = (s: number) => Math.log10(s);
  const tx = (t: number) => PAD.l + t * (size.w - PAD.l - PAD.r);
  const ty = (s: number) => PAD.t + (1 - (lg(s) - lg(0.1)) / (lg(10) - lg(0.1))) * (size.h - PAD.t - PAD.b);
  const xt = (x: number) => Math.max(0, Math.min(1, (x - PAD.l) / (size.w - PAD.l - PAD.r)));
  const ys = (y: number) => Math.pow(10, lg(0.1) + (1 - (y - PAD.t) / (size.h - PAD.t - PAD.b)) * (lg(10) - lg(0.1)));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = prepCanvas(canvas, size);
    ctx.fillStyle = THEME.bg1;
    ctx.fillRect(0, 0, size.w, size.h);
    ctx.font = '10px ui-monospace, Consolas, monospace';
    for (const s of [0.1, 0.2, 0.5, 1, 2, 5, 10]) {
      const y = Math.round(ty(s)) + 0.5;
      ctx.strokeStyle = s === 1 ? THEME.lineStrong : THEME.line;
      ctx.beginPath();
      ctx.moveTo(PAD.l, y);
      ctx.lineTo(size.w - PAD.r, y);
      ctx.stroke();
      ctx.fillStyle = THEME.textFaint;
      ctx.fillText(`${s}x`, 6, y + 3);
    }
    if (!curve) {
      ctx.fillStyle = THEME.textFaint;
      ctx.fillText('Select a clip to edit its speed ramp', PAD.l + 8, size.h / 2);
      return;
    }
    ctx.fillStyle = THEME.accentSoft;
    ctx.fillRect(PAD.l, ty(1), size.w - PAD.l - PAD.r, ty(0.1) - ty(1));
    ctx.strokeStyle = THEME.speedCurve;
    ctx.lineWidth = 2;
    ctx.beginPath();
    const N = 240;
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      const s = speedAt(curve.points, t);
      if (i === 0) ctx.moveTo(tx(t), ty(s));
      else ctx.lineTo(tx(t), ty(s));
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = THEME.speedPoint;
    curve.points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(tx(p.t), ty(p.speed), 5, 0, Math.PI * 2);
      ctx.fill();
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curve, size]);

  const hit = (x: number, y: number): number | null => {
    if (!curve) return null;
    for (let i = 0; i < curve.points.length; i++) {
      const p = curve.points[i];
      if (Math.hypot(tx(p.t) - x, ty(p.speed) - y) < 8) return i;
    }
    return null;
  };

  const pos = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!clip || !curve) return;
    const { x, y } = pos(e);
    const st = useEditor.getState();
    const i = hit(x, y);
    if (i != null) {
      if ((e.button === 2 || e.altKey) && curve.points.length > 2) {
        st.setClipSpeed(clip.id, { ...curve, preset: 'custom', points: curve.points.filter((_, j) => j !== i) });
        return;
      }
      canvasRef.current!.setPointerCapture(e.pointerId);
      st.beginGesture();
      dragRef.current = { index: i, orig: curve.points };
      return;
    }
    if (x > PAD.l && e.button === 0) {
      canvasRef.current!.setPointerCapture(e.pointerId);
      st.beginGesture();
      const added = { t: xt(x), speed: Math.max(0.1, Math.min(10, ys(y))) };
      const pts = [...curve.points, added].sort((a, b) => a.t - b.t);
      st.setClipSpeed(clip.id, { ...curve, preset: 'custom', points: pts });
      dragRef.current = { index: pts.indexOf(added), orig: pts };
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d || !clip) return;
    const cur = useEditor.getState();
    const now = cur.project.tracks.flatMap((t) => t.clips).find((c) => c.id === clip.id);
    if (!now) return;
    const { x, y } = pos(e);
    const n = d.orig.length;
    const i = d.index;
    const lo = i > 0 ? d.orig[i - 1].t + 0.005 : 0;
    const hi = i < n - 1 ? d.orig[i + 1].t - 0.005 : 1;
    const t = i === 0 ? 0 : i === n - 1 ? 1 : Math.max(lo, Math.min(hi, xt(x)));
    const pts = d.orig.map((p, j) => (j === i ? { t, speed: Math.max(0.1, Math.min(10, ys(y))) } : p));
    cur.setClipSpeed(clip.id, { ...now.speed, preset: 'custom', points: pts });
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    if (canvasRef.current?.hasPointerCapture(e.pointerId)) canvasRef.current.releasePointerCapture(e.pointerId);
    useEditor.getState().endGesture();
  };

  const srcDur = clip ? clip.outMs - clip.inMs : 0;
  const outDur = curve ? outputDuration(curve, srcDur) : 0;

  return (
    <div className="drawer-body" style={{ gridTemplateColumns: '180px 1fr' }}>
      <div className="drawer-side">
        <div className="chips">
          {(Object.keys(SPEED_PRESETS) as SpeedPreset[]).map((p) => (
            <button key={p} className={`small ${curve?.preset === p ? 'active' : ''}`} disabled={!clip} onClick={() => clip && useEditor.getState().setClipSpeed(clip.id, presetCurve(p, curve?.opticalFlow ?? true))}>
              {p.replace('_', ' ')}
            </button>
          ))}
        </div>
        {curve && clip ? (
          <>
            <label className="check-row" style={{ marginTop: 8 }}>
              <input type="checkbox" checked={curve.opticalFlow} onChange={(e) => useEditor.getState().setClipSpeed(clip.id, { ...curve, opticalFlow: e.target.checked })} />
              RAFT optical-flow slow-mo
            </label>
            <div className="hint" style={{ marginTop: 6 }}>
              Source {(srcDur / 1000).toFixed(2)}s → output {(outDur / 1000).toFixed(2)}s
              <br />
              Click to add a point · drag · Alt-click removes
            </div>
          </>
        ) : null}
      </div>
      <div ref={wrapRef} style={{ minHeight: 0, minWidth: 0 }}>
        <canvas ref={canvasRef} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onContextMenu={(e) => e.preventDefault()} />
      </div>
    </div>
  );
}
