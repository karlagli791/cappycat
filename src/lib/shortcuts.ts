/** Keyboard shortcuts (CapCut conventions where they exist). Shown by the "?" cheat sheet. */
export const SHORTCUTS: Array<{ group: string; items: Array<[keys: string, what: string]> }> = [
  {
    group: 'Playback',
    items: [
      ['Space', 'Play / pause (restarts at the end)'],
      ['J / K / L', 'Shuttle reverse / pause / forward (press again: 2x, 4x)'],
      ['← / →', 'Previous / next frame (Shift: 10 frames)'],
      ['↑ / ↓', 'Previous / next cut'],
      ['Home / End', 'Start / end of the timeline'],
    ],
  },
  {
    group: 'Edit',
    items: [
      ['Ctrl+B or S', 'Split at the playhead (selected clips, or every clip under it)'],
      ['Delete', 'Delete the selection (the magnet closes the gap), or the selected cut’s transition'],
      ['Alt+F', 'Freeze frame (1 s) at the playhead'],
      ['Alt+R', 'Reverse the selected clip'],
      ['Ctrl+Alt+C / Ctrl+Alt+V', 'Copy / paste attributes (grade, speed, motion, audio, mask)'],
      ['Ctrl+A', 'Select every clip'],
      ['Shift/Ctrl+click, drag on empty area', 'Add to selection, marquee select'],
      ['Ctrl+Z / Ctrl+Y', 'Undo / redo (Ctrl+Shift+Z also redoes)'],
      ['Esc', 'Cancel a drag, close dialogs / menus / graph, clear the selection'],
      ['Right-click a clip', 'Clip menu: split, delete, speed, separate to tracks, add transition, attributes, freeze, reverse'],
    ],
  },
  {
    group: 'Transitions, effects & audio',
    items: [
      ['Click a cut’s bow-tie', 'Select the cut (Inspector → Transition); drag a transition onto a cut to apply it'],
      ['Drag a clip corner handle', 'Fade in / out (video: top corners; audio clips: top corners)'],
      ['Drag the volume line', 'Clip volume (audio clips)'],
      ['Alt+click the volume line', 'Add a volume keyframe (Alt+click a keyframe removes it; drag it up / down)'],
      ['Effects tab: click', 'Add the effect at the playhead (or drag it onto the FX track)'],
    ],
  },
  {
    group: 'Timeline',
    items: [
      ['N', 'Snapping on/off'],
      ['P', 'Main-track magnet (ripple) on/off'],
      ['Shift+Z', 'Zoom to fit'],
      ['Ctrl+= / Ctrl+-', 'Zoom in / out (Ctrl+wheel zooms at the cursor)'],
      ['Alt+K', 'Keyframe / speed graph'],
    ],
  },
  {
    group: 'View & file',
    items: [
      ['C', 'Raw vs graded compare'],
      ['U', 'Universal adjust on/off'],
      ['F', 'Large preview (Esc exits)'],
      ['Ctrl+S / Ctrl+Shift+S', 'Save / save as'],
      ['Ctrl+O / Ctrl+N', 'Open / new project'],
      ['Ctrl+I / Ctrl+E', 'Import media / export'],
      ['?', 'This cheat sheet'],
    ],
  },
];

/** Is the keyboard focus in a control that types text (shortcuts must not fire there)? */
export function isTextEntry(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el.isContentEditable) return true;
  if (el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
  if (el.tagName !== 'INPUT') return false;
  const type = ((el as HTMLInputElement).type || 'text').toLowerCase();
  return ['text', 'number', 'search', 'email', 'password', 'url', 'tel', 'date', 'time'].includes(type);
}
