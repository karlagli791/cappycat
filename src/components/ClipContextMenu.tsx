/**
 * Right-click menu of a timeline clip: Split, Delete, Speed ▸, Separate to tracks, Add transition ▸,
 * Copy / Paste attributes, Freeze frame, Reverse. Closes on Escape, a click outside or an action.
 */
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { clipDurationMs, clipEndMs, findClip, useEditor } from '@/state/store';
import { frameMs, nextAdjacent, prevAdjacent } from '@/state/edits';
import { playClock } from '@/state/clock';
import { QUICK_SPEEDS } from '@/engine/defaults';
import { TRANSITION_DEFAULT_MS, TRANSITION_TYPES } from '@/engine/transitions';
import { averageSpeed } from '@/engine/speed';
import { canSeparateToTracks, hasStemClips } from '@/state/stemTracks';
import { separateClipsToTracks } from '@/state/separation';
import { useAiReady } from '@/state/setup';

interface Props {
  x: number;
  y: number;
  clipId: string;
  /** timeline time under the pointer (split falls back to it when the playhead is outside the clip) */
  atMs: number;
  onClose: () => void;
}

export const PASTE_ATTRIBUTES_EVENT = 'cappycat:paste-attributes';

export default function ClipContextMenu({ x, y, clipId, atMs, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [sub, setSub] = useState<'speed' | 'transition' | null>(null);
  const [pos, setPos] = useState({ left: x, top: y });
  const project = useEditor((s) => s.project);
  const hasClipboard = useEditor((s) => !!s.attrClipboard);
  const ai = useAiReady();
  const found = findClip(project, clipId);

  useLayoutEffect(() => {
    // keep the menu on screen
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos({ left: Math.min(x, window.innerWidth - r.width - 8), top: Math.min(y, window.innerHeight - r.height - 8) });
  }, [x, y]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopImmediatePropagation();
        onClose();
      }
    };
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('mousedown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('mousedown', onDown, true);
    };
  }, [onClose]);

  if (!found) return null;
  const { clip, track } = found;
  const st = useEditor.getState;
  const run = (fn: () => void) => () => {
    onClose();
    fn();
  };
  const head = playClock.get();
  const inside = (t: number) => t > clip.startMs + 20 && t < clipEndMs(clip) - 20;
  const splitAt = inside(head) ? head : atMs;
  const isFx = !!clip.effect;
  const isVideo = track.kind === 'video';
  const frame = frameMs(project);
  // "Add transition" goes on the clip's incoming cut, else on its outgoing cut
  const prev = isVideo ? prevAdjacent(track, clip, frame) : null;
  const next = isVideo ? nextAdjacent(track, clip, frame) : null;
  const transitionTarget = prev ? clip : next;
  const separable = canSeparateToTracks(project, clip.id) && !hasStemClips(project, clip.id);
  const speed = averageSpeed(clip.speed);

  return (
    <div className="context-menu" ref={ref} style={{ left: pos.left, top: pos.top }} role="menu" onContextMenu={(e) => e.preventDefault()}>
      <Item disabled={!inside(splitAt)} onClick={run(() => st().splitClipAt(clip.id, splitAt))} kbd="Ctrl+B">
        Split
      </Item>
      <Item onClick={run(() => st().deleteClips([clip.id]))} kbd="Del">
        Delete
      </Item>
      <div className="menu-sep" />
      <SubItem label="Speed" open={sub === 'speed'} onOpen={() => setSub('speed')} disabled={isFx || track.locked}>
        {QUICK_SPEEDS.map((v) => (
          <Item key={v} onClick={run(() => st().setConstantSpeed([clip.id], v))} active={Math.abs(speed - v) < 0.005}>
            {v}×
          </Item>
        ))}
      </SubItem>
      <SubItem label="Add transition" open={sub === 'transition'} onOpen={() => setSub('transition')} disabled={!transitionTarget || track.locked}>
        {transitionTarget
          ? TRANSITION_TYPES.map((t) => (
              <Item
                key={t.id}
                title={t.hint}
                active={transitionTarget.transitionIn?.type === t.id}
                onClick={run(() => {
                  st().setTransition(transitionTarget.id, { type: t.id, durationMs: transitionTarget.transitionIn?.durationMs ?? TRANSITION_DEFAULT_MS });
                  st().selectCut(transitionTarget.id);
                })}
              >
                {t.label}
              </Item>
            ))
          : null}
      </SubItem>
      <Item
        disabled={!separable || !ai.ready}
        title={!ai.ready ? ai.hint : separable ? 'Voice and background stems as their own clips on the Voice / Background tracks' : 'No sound to separate, or already separated'}
        onClick={run(() => void separateClipsToTracks([clip.id]))}
      >
        Separate to tracks
      </Item>
      <div className="menu-sep" />
      <Item onClick={run(() => st().copyAttributes(clip.id) && st().log('info', 'Attributes copied: paste them with Ctrl+Alt+V'))} kbd="Ctrl+Alt+C">
        Copy attributes
      </Item>
      <Item disabled={!hasClipboard} onClick={run(() => window.dispatchEvent(new CustomEvent(PASTE_ATTRIBUTES_EVENT)))} kbd="Ctrl+Alt+V">
        Paste attributes…
      </Item>
      <div className="menu-sep" />
      <Item disabled={isFx || track.locked || !inside(head)} title={inside(head) ? 'Hold the frame under the playhead for 1 s' : 'Move the playhead into the clip first'} onClick={run(() => st().freezeFrameAt(clip.id, head))} kbd="Alt+F">
        Freeze frame
      </Item>
      <Item disabled={isFx || track.locked} onClick={run(() => st().toggleReverse(clip.id))} kbd="Alt+R">
        {clip.reversed ? 'Play forward' : 'Reverse'}
      </Item>
      <div className="context-meta">
        {clip.label ?? 'clip'} · {(clipDurationMs(clip) / 1000).toFixed(2)} s
      </div>
    </div>
  );
}

function Item({ children, onClick, disabled, kbd, title, active }: { children: React.ReactNode; onClick: () => void; disabled?: boolean; kbd?: string; title?: string; active?: boolean }) {
  return (
    <button className={`ghost menu-item ${active ? 'active' : ''}`} role="menuitem" disabled={disabled} onClick={onClick} title={title}>
      <span className="menu-label">{children}</span>
      {kbd ? <span className="kbd">{kbd}</span> : null}
    </button>
  );
}

function SubItem({ label, open, onOpen, disabled, children }: { label: string; open: boolean; onOpen: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <div className="menu-sub" onMouseEnter={disabled ? undefined : onOpen}>
      <button className={`ghost menu-item ${open ? 'active' : ''}`} role="menuitem" aria-haspopup="menu" aria-expanded={open} disabled={disabled} onClick={onOpen}>
        <span className="menu-label">{label}</span>
        <span className="menu-arrow">▸</span>
      </button>
      {open && !disabled ? (
        <div className="menu-pop sub" role="menu">
          {children}
        </div>
      ) : null}
    </div>
  );
}
