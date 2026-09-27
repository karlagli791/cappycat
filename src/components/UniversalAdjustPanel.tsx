import { useState } from 'react';
import { useEditor } from '@/state/store';
import { api } from '@/lib/tauri';
import { HSL_CHANNELS } from '@/engine/defaults';
import type { AdjustValues, HslChannel, HslOffset } from '@/types/project';
import { Slider } from './ColorPanel';

const ROWS: Array<{ key: Exclude<keyof AdjustValues, 'hsl'>; label: string; min: number; max: number }> = [
  { key: 'exposure', label: 'Exposure', min: -50, max: 50 },
  { key: 'brilliance', label: 'Brilliance', min: -50, max: 50 },
  { key: 'contrast', label: 'Contrast', min: -50, max: 50 },
  { key: 'brightness', label: 'Lightness', min: -50, max: 50 },
  { key: 'highlights', label: 'Highlights', min: -50, max: 50 },
  { key: 'shadows', label: 'Shadows', min: -50, max: 50 },
  { key: 'saturation', label: 'Saturation', min: -50, max: 50 },
  { key: 'vibrance', label: 'Vibrance', min: -50, max: 50 },
  { key: 'sharpness', label: 'Sharpness', min: 0, max: 50 },
  { key: 'temperature', label: 'Temperature', min: -50, max: 50 },
  { key: 'tint', label: 'Tint', min: -50, max: 50 },
];

const DOT: Record<HslChannel, string> = {
  red: '#ff4d4d',
  orange: '#ff9a3c',
  yellow: '#ffe14d',
  green: '#4dff70',
  cyan: '#4df2ff',
  blue: '#4d9dff',
  purple: '#a64dff',
  magenta: '#ff4dd2',
};

const zero = (): HslOffset => ({ h: 0, s: 0, l: 0 });

export default function UniversalAdjustPanel() {
  const ua = useEditor((s) => s.project.universalAdjust);
  const preset = useEditor((s) => s.universalPreset);
  const presetPath = useEditor((s) => s.universalPresetPath);
  const setEnabled = useEditor((s) => s.setUniversalEnabled);
  const setValues = useEditor((s) => s.setUniversalValues);
  const resetToPreset = useEditor((s) => s.resetUniversalToPreset);
  const setUniversalPreset = useEditor((s) => s.setUniversalPreset);
  const log = useEditor((s) => s.log);
  const [channel, setChannel] = useState<HslChannel>('red');
  const [saved, setSaved] = useState<string | null>(null);

  const values = ua?.values ?? {};
  const enabled = !!ua?.enabled;
  const hsl = values.hsl ?? ({} as Record<HslChannel, HslOffset>);
  const cur = hsl[channel] ?? zero();
  const differsFromPreset = !!preset && JSON.stringify(preset.values) !== JSON.stringify(values);

  const setHsl = (patch: Partial<HslOffset>) => {
    const full = {} as Record<HslChannel, HslOffset>;
    for (const ch of HSL_CHANNELS) full[ch] = { ...(hsl[ch] ?? zero()) };
    full[channel] = { ...full[channel], ...patch };
    setValues({ ...values, hsl: full });
  };

  const save = async () => {
    const file = { version: 1, name: ua?.name ?? 'Universal adjust', enabledByDefault: preset?.enabledByDefault ?? true, values };
    try {
      const res = await api.saveUniversalAdjust(file);
      setUniversalPreset(res.preset, res.path);
      setSaved(`Saved to ${res.path}`);
      log('info', `Universal adjust preset saved to ${res.path}`);
    } catch (e) {
      setSaved(`Save failed: ${String(e)}`);
    }
  };

  const toggleDefault = async () => {
    if (!preset) return;
    try {
      const res = await api.saveUniversalAdjust({ ...preset, enabledByDefault: !preset.enabledByDefault });
      setUniversalPreset(res.preset, res.path);
    } catch (e) {
      setSaved(`Could not update the preset: ${String(e)}`);
      log('error', `Universal adjust preset not saved: ${String(e)}`);
    }
  };

  return (
    <div className="section">
      <h4>
        Universal adjust · every clip
        <label className="check-row" style={{ textTransform: 'none', letterSpacing: 0, margin: 0 }}>
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
          {enabled ? 'On' : 'Off'}
        </label>
      </h4>
      <div className="hint" style={{ marginBottom: 6 }}>
        Added on top of each clip's own grade, in the preview and in exports. Toggle with <span className="kbd">U</span>.
      </div>
      <div style={{ opacity: enabled ? 1 : 0.5 }}>
        {ROWS.map((r) => (
          <Slider
            key={r.key}
            label={r.label}
            value={values[r.key] ?? 0}
            min={r.min}
            max={r.max}
            onChange={(v) => setValues({ ...values, [r.key]: v })}
          />
        ))}
        <h4 style={{ marginTop: 10 }}>HSL</h4>
        <div className="hsl-dots">
          {HSL_CHANNELS.map((ch) => {
            const o = hsl[ch];
            const touched = !!o && (o.h !== 0 || o.s !== 0 || o.l !== 0);
            return (
              <button
                key={ch}
                className={`hsl-dot ${channel === ch ? 'selected' : ''}`}
                style={{ borderColor: DOT[ch], background: channel === ch || touched ? DOT[ch] : 'transparent' }}
                onClick={() => setChannel(ch)}
                title={ch}
              />
            );
          })}
        </div>
        <Slider label="Hue" value={cur.h} min={-100} max={100} onChange={(v) => setHsl({ h: v })} />
        <Slider label="Saturation" value={cur.s} min={-100} max={100} onChange={(v) => setHsl({ s: v })} />
        <Slider label="Brightness" value={cur.l} min={-100} max={100} onChange={(v) => setHsl({ l: v })} />
      </div>
      <div className="chips" style={{ marginTop: 8 }}>
        <button className="small primary" onClick={save} title="Make these values the universal preset for every project">
          Save as universal preset
        </button>
        <button className="small" onClick={resetToPreset} disabled={!differsFromPreset} title="Discard edits and use the saved preset">
          Revert to saved
        </button>
      </div>
      {preset ? (
        <label className="check-row" style={{ marginTop: 8 }}>
          <input type="checkbox" checked={preset.enabledByDefault} onChange={() => void toggleDefault()} />
          Switch on automatically for new projects and renders
        </label>
      ) : null}
      <div className="hint" style={{ marginTop: 4 }}>
        {saved ?? (presetPath ? `Preset file: ${presetPath}` : '')}
        {differsFromPreset ? ' · unsaved changes' : ''}
      </div>
    </div>
  );
}
