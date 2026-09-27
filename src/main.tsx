import React, { Profiler } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';
import { perfStats } from './dev/perfStats';

if (import.meta.env.DEV) {
  // ?perf: drive requestAnimationFrame from a 60 Hz timer so the perf harness also runs in a hidden pane
  if (new URLSearchParams(location.search).has('perf')) installTimerRaf();
  void import('./dev/perfHarness').then((m) => m.installPerfHarness());
}

function installTimerRaf() {
  const pending = new Map<number, FrameRequestCallback>();
  let nextId = 1;
  let next = performance.now();
  const pump = () => {
    const now = performance.now();
    if (now >= next) {
      next = Math.max(next + 1000 / 60, now);
      const cbs = [...pending.values()];
      pending.clear();
      cbs.forEach((cb) => cb(now));
    }
    setTimeout(pump, 0);
  };
  setTimeout(pump, 0);
  window.requestAnimationFrame = (cb) => {
    const id = nextId++;
    pending.set(id, cb);
    return id;
  };
  window.cancelAnimationFrame = (id) => void pending.delete(id);
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    {import.meta.env.DEV ? (
      <Profiler id="app" onRender={() => void perfStats.commits++}>
        <App />
      </Profiler>
    ) : (
      <App />
    )}
  </React.StrictMode>,
);
