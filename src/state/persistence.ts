/**
 * Saving, opening, autosave and the unsaved-changes guard.
 *
 * Autosave: every 60 s while the document is dirty, `save_project` writes
 * `<cacheDir>/autosave/<projectId>.json` (browser mode: local storage). An index in local storage
 * remembers when each project was autosaved and last saved explicitly; on start, an autosave newer
 * than the last explicit save is offered for restore (non-blocking banner).
 */
import { api, isTauri } from '@/lib/tauri';
import { basename } from '@/lib/format';
import type { Project } from '@/types/project';
import { ask } from '@/components/Dialogs';
import { useEditor } from './store';

export const AUTOSAVE_INTERVAL_MS = 60_000;
const INDEX_KEY = 'cappycat:autosave-index';
const SAVED_KEY = 'cappycat:saved-at';

export interface AutosaveEntry {
  projectId: string;
  name: string;
  /** autosave file (native) or local-storage key (browser) */
  path: string;
  /** where the project was saved explicitly, if anywhere */
  projectPath: string | null;
  at: number;
}

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, v: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* storage unavailable */
  }
}

function autosaveIndex(): Record<string, AutosaveEntry> {
  return readJson<Record<string, AutosaveEntry>>(INDEX_KEY, {});
}

function savedAt(): Record<string, number> {
  return readJson<Record<string, number>>(SAVED_KEY, {});
}

let cacheDir: string | null = null;

export function setAutosaveDir(dir: string | null): void {
  cacheDir = dir && dir !== '(browser)' ? dir : null;
}

function autosavePath(projectId: string): string {
  if (isTauri() && cacheDir) return `${cacheDir.replace(/[\\/]+$/, '')}/autosave/${projectId}.json`;
  return `cappycat:autosave:${projectId}`;
}

function hasContent(p: Project): boolean {
  return p.assets.length > 0 || p.tracks.some((t) => t.clips.length > 0);
}

/** Write the autosave now (if dirty). Returns true when written. */
export async function autosaveNow(): Promise<boolean> {
  const s = useEditor.getState();
  if (!s.dirty || !hasContent(s.project)) return false;
  if (isTauri() && !cacheDir) return false;
  const p = s.project;
  const path = autosavePath(p.id);
  try {
    if (isTauri()) await api.saveProject(path, p);
    else writeJson(path, p);
    const at = Date.now();
    // keep the ten most recent autosaves
    const entries = Object.values({ ...autosaveIndex(), [p.id]: { projectId: p.id, name: p.name, path, projectPath: s.projectPath, at } }).sort((a, b) => b.at - a.at);
    for (const old of entries.slice(10)) discardAutosave(old.projectId);
    writeJson(INDEX_KEY, Object.fromEntries(entries.slice(0, 10).map((e) => [e.projectId, e])));
    useEditor.getState().setSaveStatus({ autosavedAt: at });
    return true;
  } catch (e) {
    useEditor.getState().log('warn', `Autosave failed: ${String(e)}`);
    return false;
  }
}

/** Start the 60 s autosave timer; returns a stop function. */
export function startAutosave(): () => void {
  const id = setInterval(() => void autosaveNow(), AUTOSAVE_INTERVAL_MS);
  return () => clearInterval(id);
}

/** The newest autosave that is newer than its project's last explicit save, if any. */
export function pendingRestore(): AutosaveEntry | null {
  const saved = savedAt();
  const entries = Object.values(autosaveIndex()).filter((e) => e && e.at > (saved[e.projectId] ?? 0));
  entries.sort((a, b) => b.at - a.at);
  return entries[0] ?? null;
}

export function discardAutosave(projectId: string): void {
  const idx = autosaveIndex();
  const e = idx[projectId];
  delete idx[projectId];
  writeJson(INDEX_KEY, idx);
  if (e && !isTauri()) {
    try {
      localStorage.removeItem(e.path);
    } catch {
      /* ignore */
    }
  }
}

export async function restoreAutosave(e: AutosaveEntry): Promise<boolean> {
  try {
    let project: Project;
    if (isTauri()) project = (await api.openDocument(e.path)).project;
    else {
      const raw = localStorage.getItem(e.path);
      if (!raw) throw new Error('autosave data missing');
      project = JSON.parse(raw) as Project;
    }
    useEditor.getState().loadDocument(project, { path: e.projectPath, dirty: true });
    useEditor.getState().log('info', `Restored the autosave of "${e.name}" from ${new Date(e.at).toLocaleString()}`);
    return true;
  } catch (err) {
    useEditor.getState().log('error', `Could not restore the autosave: ${String(err)}`);
    discardAutosave(e.projectId);
    return false;
  }
}

/** Save (Ctrl+S) or Save as (Ctrl+Shift+S). Returns true when the project was written. */
export async function saveProject(as = false): Promise<boolean> {
  const s = useEditor.getState();
  let path = s.projectPath;
  if (!path || as) path = await api.pickSavePath(s.project.name || 'project', 'cappycat.json');
  if (!path) return false;
  const project = useEditor.getState().project;
  useEditor.getState().setSaveStatus({ state: 'saving', message: undefined });
  try {
    await api.saveProject(path, project);
    const st = useEditor.getState();
    st.setProjectPath(path);
    // only clean if nothing changed while the file was being written
    if (st.project === project) st.markSaved();
    else st.setSaveStatus({ state: 'saved', savedAt: Date.now() });
    writeJson(SAVED_KEY, { ...savedAt(), [project.id]: Date.now() });
    discardAutosave(project.id);
    api.addRecentProject(path);
    st.log('info', `Saved ${basename(path)}`);
    return true;
  } catch (e) {
    useEditor.getState().setSaveStatus({ state: 'error', message: String(e) });
    useEditor.getState().log('error', `Save failed: ${String(e)}`);
    return false;
  }
}

/**
 * Ask what to do with unsaved changes before `action` (New / Open / Close).
 * Returns true when it is fine to continue.
 */
export async function guardUnsaved(action: string): Promise<boolean> {
  const s = useEditor.getState();
  if (!s.dirty) return true;
  const choice = await ask({
    title: `Save changes to "${s.project.name}"?`,
    message: `Your changes will be lost if you ${action} without saving.`,
    buttons: [
      { id: 'cancel', label: 'Cancel' },
      { id: 'discard', label: "Don't save", kind: 'danger' },
      { id: 'save', label: 'Save', kind: 'primary' },
    ],
  });
  if (choice === 'save') return saveProject();
  return choice === 'discard';
}

export async function newProject(): Promise<void> {
  if (!(await guardUnsaved('start a new project'))) return;
  useEditor.getState().newProject('Untitled');
}

/** File > Open (project or pipeline analysis), or a recent file. */
export async function openProject(path?: string): Promise<void> {
  if (!(await guardUnsaved('open another project'))) return;
  const file = path ?? (await api.pickOpenPath('json'));
  if (!file) return;
  const st = useEditor.getState();
  try {
    const doc = await api.openDocument(file);
    if (doc.kind === 'analysis' && doc.analysis) {
      // an analysis opens as its assembled timeline plus the findings (save it as a project)
      st.loadDocument(doc.project, { path: null, analysis: doc.analysis });
      st.log('info', `Opened analysis ${basename(file)}`);
    } else {
      st.loadDocument(doc.project, { path: file });
      api.addRecentProject(file);
      st.log('info', `Opened ${basename(file)}`);
    }
    useEditor.getState().setSaveStatus({ state: 'idle', savedAt: null });
  } catch (e) {
    st.log('error', `Open failed: ${String(e)}`);
  }
}
