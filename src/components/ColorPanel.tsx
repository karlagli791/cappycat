import { useEffect, useRef, useState } from 'react';
import { useEditor } from '@/state/store';
import { COLOR_PRESETS, HSL_CHANNELS, defaultColorGrade } from '@/engine/defaults';
import { evalCurve } from '@/engine/color/curves';
import type { Asset, Clip, ColorGrade, CurvePoint, HslChannel, Vec3 } from '@/types/project';
import { THEME } from '@/lib/theme';
import UniversalAdjustPanel from './UniversalAdjustPanel';
import { confirmAction } from './Dialogs';

const HSL_COLORS: Record<HslChannel, string> = {
  red: '#ff4d4d',
  orange: '#ff9a3c',
  yellow: '#ffe14d',
  green: '#4dff70',
  cyan: '#4df2ff',
  blue: '#4d7dff',
  purple: '#a64dff',
  magenta: '#ff4dd2',
};

/**
 * Pointer-down handler for any continuous control: the whole drag becomes one undo step
 * (store gesture, ended on the next pointer-up anywhere).
 */
export function beginControlGesture(): void {
  const st = useEditor.getState();
  st.beginGesture();
  const end = () => {
    window.removeEventListener('pointerup', end, true);
    window.removeEventListener('pointercancel', end, true);
    useEditor.getState().endGesture();
  };
  window.addEventListener('pointerup', end, true);
  window.addEventListener('pointercancel', end, true);
}

export default function ColorPanel({ clip, assets }: { clip: Clip; assets: Asset[] }) {
  const update = useEditor((s) => s.updateClipColor);
  const g = clip.color;
  const set = (patch: Partial<ColorGrade>) => update(clip.id, patch);
  const luts = assets.filter((a) => a.kind === 'lut');

  const applyAll = async (sameSource: boolean) => {
    const st = useEditor.getState();
    const ok = await confirmAction(
      sameSource ? 'Apply this grade to the same source?' : 'Apply this grade to every clip?',
      sameSource ? 'Every other clip cut from this file gets this clip’s grade (LUT, curves, wheels, HSL).' : 'Every video clip on the timeline gets this clip’s grade (LUT, curves, wheels, HSL). Undo with Ctrl+Z.',
      'Apply',
    );
    if (!ok) return;
    const n = st.applyGradeToAll(clip.id, sameSource);
    st.log('info', `Grade applied to ${n} clip(s)`);
  };

  return (
    <div>
      <UniversalAdjustPanel />
      <div className="section">
        <h4 style={{ color: 'var(--text)' }}>This clip</h4>
        <div className="hint">These values add to the universal adjust above.</div>
        <div className="chips" style={{ marginTop: 6 }}>
          <button className="small" onClick={() => void applyAll(false)} title="Copy this grade to every video clip">
            Apply to all clips
          </button>
          <button className="small" onClick={() => void applyAll(true)} title="Copy this grade to the other clips cut from the same file">
            Apply to same source
          </button>
        </div>
      </div>
      <div className="section">
        <h4>
          Presets
          <button
            className="small ghost"
            onClick={async () => {
              if (await confirmAction('Reset the grade?', 'All color settings of this clip go back to neutral (undo with Ctrl+Z).', 'Reset', true)) set(defaultColorGrade());
            }}
          >
            Reset
          </button>
        </h4>
        <div className="chips">
          {Object.entries(COLOR_PRESETS).map(([name, preset]) => (
            <button key={name} className="small" onClick={() => set({ ...defaultColorGrade(), ...preset, lutAssetId: g.lutAssetId, lutIntensity: g.lutIntensity })}>
              {name.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      <div className="section">
        <h4>Primary adjustments</h4>
        {(
          [
            ['exposure', 'Exposure'],
            ['brilliance', 'Brilliance'],
            ['contrast', 'Contrast'],
            ['brightness', 'Lightness'],
            ['highlights', 'Highlights'],
            ['shadows', 'Shadows'],
            ['saturation', 'Saturation'],
            ['vibrance', 'Vibrance'],
            ['sharpness', 'Sharpness'],
          ] as Array<[keyof ColorGrade, string]>
        ).map(([key, label]) => (
          <Slider key={key} label={label} value={(g[key] as number | undefined) ?? 0} min={key === 'sharpness' ? 0 : -50} max={50} onChange={(v) => set({ [key]: v })} />
        ))}
      </div>

      <div className="section">
        <h4>White balance</h4>
        <Slider label="Temperature" value={g.temperature} min={-50} max={50} onChange={(v) => set({ temperature: v })} hint="blue ↔ yellow" />
        <Slider label="Tint" value={g.tint} min={-50} max={50} onChange={(v) => set({ tint: v })} hint="green ↔ magenta" />
      </div>

      <div className="section">
        <h4>3-way color wheels</h4>
        <div className="wheels">
          {(
            [
              ['lift', 'Lift'],
              ['gamma', 'Gamma'],
              ['gain', 'Gain'],
              ['offset', 'Offset'],
            ] as Array<[keyof Pick<ColorGrade, 'lift' | 'gamma' | 'gain' | 'offset'>, string]>
          ).map(([key, label]) => (
            <Wheel key={key} label={label} value={g[key]} onChange={(v) => set({ [key]: v })} />
          ))}
        </div>
      </div>

      <div className="section">
        <h4>HSL tuning</h4>
        <div className="hsl-grid">
          <span />
          <span className="hint">Hue</span>
          <span className="hint">Sat</span>
          <span className="hint">Lum</span>
          {HSL_CHANNELS.map((ch) => (
            <HslRow key={ch} ch={ch} value={g.hsl[ch]} onChange={(v) => set({ hsl: { ...g.hsl, [ch]: v } })} />
          ))}
        </div>
      </div>

      <div className="section">
        <h4>RGB curves</h4>
        <CurvesEditor curves={g.curves} onChange={(curves) => set({ curves })} />
      </div>

      <div className="section">
        <h4>3D LUT</h4>
        <div className="row">
          <label>.cube file</label>
          <select value={g.lutAssetId ?? ''} onChange={(e) => set({ lutAssetId: e.target.value || null })} style={{ gridColumn: '2 / 4' }}>
            <option value="">None</option>
            {luts.map((l) => (
              <option key={l.id} value={l.id}>
                {l.name}
              </option>
            ))}
          </select>
        </div>
        <Slider label="Intensity" value={Math.round(g.lutIntensity * 100)} min={0} max={100} defaultValue={100} onChange={(v) => set({ lutIntensity: v / 100 })} />
        {!luts.length ? <div className="hint">Import a .cube LUT (17³–64³) into the media library to use it here.</div> : null}
      </div>

      <div className="section">
        <h4>Effects</h4>
        <Slider label="Vignette" value={Math.round(g.vignette * 100)} min={0} max={100} onChange={(v) => set({ vignette: v / 100 })} />
        <Slider label="Grain" value={Math.round(g.grain * 100)} min={0} max={100} onChange={(v) => set({ grain: v / 100 })} />
      </div>
    </div>
  );
}

/**
 * Labelled range slider. A drag is one undo step; double-click the label to reset; click the value
 * to type an exact number (Enter applies, Esc cancels).
 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  hint,
  fmt,
  defaultValue,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  hint?: string;
  fmt?: (v: number) => string;
  defaultValue?: number;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const reset = defaultValue ?? (min < 0 ? 0 : min);
  const shown = fmt ? fmt(value) : Number.isInteger(step) ? String(value) : value.toFixed(2);
  const commitTyped = () => {
    if (editing == null) return;
    const v = Number(editing.replace(/[^\d.+-]/g, ''));
    setEditing(null);
    if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, Math.round(v / step) * step)));
  };
  return (
    <div className="row" title={hint}>
      <label onDoubleClick={() => onChange(reset)} title={`${hint ? `${hint} · ` : ''}Double-click to reset`}>
        {label}
      </label>
      <input type="range" min={min} max={max} step={step} value={value} onPointerDown={beginControlGesture} onChange={(e) => onChange(Number(e.target.value))} aria-label={label} />
      {editing != null ? (
        <input
          className="value-input"
          type="number"
          autoFocus
          value={editing}
          min={min}
          max={max}
          step={step}
          onChange={(e) => setEditing(e.target.value)}
          onBlur={commitTyped}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitTyped();
            if (e.key === 'Escape') {
              e.stopPropagation();
              setEditing(null);
            }
          }}
        />
      ) : (
        <button className="value" onClick={() => setEditing(String(Number(value.toFixed(3))))} title="Click to type a value">
          {shown}
        </button>
      )}
    </div>
  );
}

/** Colour wheel: drag the puck to push the balance towards a hue; sliders for R/G/B and master. */
function Wheel({ label, value, onChange }: { label: string; value: Vec3; onChange: (v: Vec3) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const master = (value[0] + value[1] + value[2]) / 3;
  // chroma vector (red up, green at 120°, blue at 240°, clockwise like the conic gradient)
  const PHI = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
  const MAX = 0.25;
  let vx = 0;
  let vy = 0;
  PHI.forEach((p, i) => {
    vx += (value[i] - master) * Math.sin(p);
    vy += (value[i] - master) * Math.cos(p);
  });
  const amp = Math.min(1, ((2 / 3) * Math.hypot(vx, vy)) / MAX);
  const ang = Math.atan2(vx, vy);
  const setC = (i: number, v: number) => {
    const next: Vec3 = [...value] as Vec3;
    next[i] = v;
    onChange(next);
  };
  const fromPointer = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    const rad = Math.min(1, Math.hypot(dx, dy) / (r.width / 2));
    const th = Math.atan2(dx, -dy);
    onChange(PHI.map((p) => master + rad * MAX * Math.cos(th - p)) as Vec3);
  };
  return (
    <div className="wheel">
      {label}
      <div
        className="swatch"
        ref={ref}
        onPointerDown={(e) => {
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          beginControlGesture();
          fromPointer(e);
        }}
        onPointerMove={(e) => {
          if ((e.currentTarget as HTMLElement).hasPointerCapture(e.pointerId)) fromPointer(e);
        }}
        onDoubleClick={() => onChange([0, 0, 0])}
        title="Drag to tint · double-click to reset"
      >
        <span className="puck" style={{ left: `${50 + 50 * amp * Math.sin(ang)}%`, top: `${50 - 50 * amp * Math.cos(ang)}%` }} />
      </div>
      {['R', 'G', 'B'].map((c, i) => (
        <input key={c} type="range" min={-0.5} max={0.5} step={0.005} value={value[i]} onPointerDown={beginControlGesture} onChange={(e) => setC(i, Number(e.target.value))} title={c} aria-label={`${label} ${c}`} />
      ))}
      <input
        type="range"
        min={-0.5}
        max={0.5}
        step={0.005}
        value={master}
        onPointerDown={beginControlGesture}
        onChange={(e) => {
          const d = Number(e.target.value) - master;
          onChange([value[0] + d, value[1] + d, value[2] + d]);
        }}
        title="Master"
        aria-label={`${label} master`}
        style={{ marginTop: 4 }}
      />
    </div>
  );
}

function HslRow({ ch, value, onChange }: { ch: HslChannel; value: { h: number; s: number; l: number }; onChange: (v: { h: number; s: number; l: number }) => void }) {
  return (
    <>
      <span className="sw">
        <i style={{ background: HSL_COLORS[ch] }} />
        {ch}
      </span>
      {(['h', 's', 'l'] as const).map((k) => (
        <input
          key={k}
          type="range"
          min={-100}
          max={100}
          value={value[k]}
          onPointerDown={beginControlGesture}
          onDoubleClick={() => onChange({ ...value, [k]: 0 })}
          onChange={(e) => onChange({ ...value, [k]: Number(e.target.value) })}
          aria-label={`${ch} ${k}`}
          title="Double-click to reset"
        />
      ))}
    </>
  );
}

type Channel = 'master' | 'r' | 'g' | 'b';
const CH_COLOR: Record<Channel, string> = { master: THEME.curveMaster, r: THEME.curveR, g: THEME.curveG, b: THEME.curveB };

function CurvesEditor({ curves, onChange }: { curves: ColorGrade['curves']; onChange: (c: ColorGrade['curves']) => void }) {
  const [ch, setCh] = useState<Channel>('master');
  const ref = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<number | null>(null);
  const size = 260;

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = size * dpr;
    c.height = size * dpr;
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = THEME.curvesBg;
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = THEME.curvesGrid;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo((size * i) / 4, 0);
      ctx.lineTo((size * i) / 4, size);
      ctx.moveTo(0, (size * i) / 4);
      ctx.lineTo(size, (size * i) / 4);
      ctx.stroke();
    }
    ctx.strokeStyle = THEME.curvesDiagonal;
    ctx.beginPath();
    ctx.moveTo(0, size);
    ctx.lineTo(size, 0);
    ctx.stroke();
    for (const k of ['r', 'g', 'b', 'master'] as Channel[]) {
      const pts = curves[k];
      ctx.strokeStyle = CH_COLOR[k];
      ctx.globalAlpha = k === ch ? 1 : 0.35;
      ctx.lineWidth = k === ch ? 2 : 1;
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const y = evalCurve(pts, x);
        if (i === 0) ctx.moveTo(x * size, (1 - y) * size);
        else ctx.lineTo(x * size, (1 - y) * size);
      }
      ctx.stroke();
      if (k === ch) {
        ctx.fillStyle = CH_COLOR[k];
        for (const [x, y] of pts) {
          ctx.beginPath();
          ctx.arc(x * size, (1 - y) * size, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
  }, [curves, ch]);

  const pos = (e: React.PointerEvent): [number, number] => {
    const r = ref.current!.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), Math.max(0, Math.min(1, 1 - (e.clientY - r.top) / r.height))];
  };
  const setPts = (pts: CurvePoint[]) => onChange({ ...curves, [ch]: pts });

  return (
    <div>
      <div className="chips" style={{ marginBottom: 6 }}>
        {(['master', 'r', 'g', 'b'] as Channel[]).map((k) => (
          <button key={k} className={`small ${ch === k ? 'active' : ''}`} style={{ color: CH_COLOR[k] }} onClick={() => setCh(k)}>
            {k.toUpperCase()}
          </button>
        ))}
        <button className="small ghost" onClick={() => setPts([[0, 0], [1, 1]])}>
          Reset
        </button>
      </div>
      <canvas
        ref={ref}
        className="curves-editor"
        onPointerDown={(e) => {
          const [x, y] = pos(e);
          const pts = curves[ch];
          const i = pts.findIndex(([px, py]) => Math.hypot(px - x, py - y) < 0.05);
          if (i >= 0 && e.altKey && i !== 0 && i !== pts.length - 1) {
            setPts(pts.filter((_, j) => j !== i));
            return;
          }
          ref.current!.setPointerCapture(e.pointerId);
          useEditor.getState().beginGesture();
          if (i >= 0) dragRef.current = i;
          else {
            const next = [...pts, [x, y] as CurvePoint].sort((a, b) => a[0] - b[0]);
            setPts(next);
            dragRef.current = next.findIndex(([px]) => px === x);
          }
        }}
        onPointerMove={(e) => {
          if (dragRef.current == null) return;
          const [x, y] = pos(e);
          const pts = curves[ch];
          const i = dragRef.current;
          if (!pts[i]) return;
          const lock = i === 0 || i === pts.length - 1;
          const nx = lock ? pts[i][0] : Math.max(pts[i - 1][0] + 0.01, Math.min(pts[i + 1][0] - 0.01, x));
          setPts(pts.map((p, j) => (j === i ? ([nx, y] as CurvePoint) : p)));
        }}
        onPointerUp={(e) => {
          if (dragRef.current == null) return;
          dragRef.current = null;
          if (ref.current?.hasPointerCapture(e.pointerId)) ref.current.releasePointerCapture(e.pointerId);
          useEditor.getState().endGesture();
        }}
        onPointerCancel={() => {
          dragRef.current = null;
          useEditor.getState().endGesture();
        }}
      />
      <div className="hint">Click to add a point, drag to shape, Alt-click to remove.</div>
    </div>
  );
}
