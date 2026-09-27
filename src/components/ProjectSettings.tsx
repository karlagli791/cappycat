/**
 * Project settings (File → Project settings…, or the project chip): output frame rate and the frame
 * interpolation the exporter uses when a clip's effective frame rate is below the output rate.
 */
import { useEffect, useState } from 'react';
import { useEditor } from '@/state/store';
import { FRAME_INTERPOLATIONS, PROJECT_FPS_OPTIONS, frameInterpolationOf } from '@/engine/defaults';
import type { FrameInterpolation } from '@/types/project';
import { CloseIcon } from './Overlays';

/** Frame rate + interpolation pickers, shared with the Export dialog. */
export function FrameRateFields({
  fps,
  interpolation,
  onFps,
  onInterpolation,
  disabled,
}: {
  fps: number;
  interpolation: FrameInterpolation;
  onFps: (v: number) => void;
  onInterpolation: (v: FrameInterpolation) => void;
  disabled?: boolean;
}) {
  const known = (PROJECT_FPS_OPTIONS as readonly number[]).includes(fps);
  return (
    <>
      <label htmlFor="fps-select">Frame rate</label>
      <select id="fps-select" value={String(fps)} onChange={(e) => onFps(Number(e.target.value))} disabled={disabled}>
        {!known ? <option value={String(fps)}>{fps} fps (source)</option> : null}
        {PROJECT_FPS_OPTIONS.map((f) => (
          <option key={f} value={String(f)}>
            {f} fps
          </option>
        ))}
      </select>
      <label htmlFor="interp-select">Frame interpolation</label>
      <select id="interp-select" value={interpolation} onChange={(e) => onInterpolation(e.target.value as FrameInterpolation)} disabled={disabled}>
        {FRAME_INTERPOLATIONS.map((f) => (
          <option key={f.id} value={f.id} title={f.hint}>
            {f.label} · {f.hint}
          </option>
        ))}
      </select>
    </>
  );
}

export const INTERPOLATION_NOTE =
  'Interpolation is applied on export, when the output frame rate is higher than a clip’s effective rate (source fps × speed): slow motion and 24 → 60 fps conversions. The preview plays the source frames.';

export default function ProjectSettings({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return <ProjectSettingsBody onClose={onClose} />;
}

function ProjectSettingsBody({ onClose }: { onClose: () => void }) {
  const project = useEditor((s) => s.project);
  const [fps, setFps] = useState(project.fps);
  const [interp, setInterp] = useState<FrameInterpolation>(frameInterpolationOf(project));
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopImmediatePropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onClose]);
  const save = () => {
    const st = useEditor.getState();
    st.setProjectSettings({ fps, frameInterpolation: interp });
    st.log('info', `Project: ${fps} fps, frame interpolation ${FRAME_INTERPOLATIONS.find((f) => f.id === interp)?.label}`);
    onClose();
  };
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal" role="dialog" aria-label="Project settings" onMouseDown={(e) => e.stopPropagation()}>
        <h3>
          Project settings
          <button className="small ghost" onClick={onClose} title="Close (Esc)" aria-label="Close">
            <CloseIcon />
          </button>
        </h3>
        <div className="export-grid">
          <label>Name</label>
          <span className="hint">{project.name}</span>
          <label>Resolution</label>
          <span className="hint">
            {project.width}×{project.height}
          </span>
          <FrameRateFields fps={fps} interpolation={interp} onFps={setFps} onInterpolation={setInterp} />
        </div>
        <div className="finding" style={{ marginTop: 12 }}>
          {INTERPOLATION_NOTE}
        </div>
        <div className="actions">
          <button onClick={onClose}>Cancel</button>
          <button className="primary" onClick={save}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
