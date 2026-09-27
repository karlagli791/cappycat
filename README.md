# Cappycat

Automated local AI video editor: ingests raw 15–50 s AI-generated clips, detects
camera cuts, finds duplicated characters, reframes them out with director-style
auto-zoom, normalises audio, and drops the result onto a CapCut-style
non-destructive timeline with full manual control (color grading, speed ramps,
keyframe graphs, masks). Everything runs on the local machine.

The design is in [docs/design-spec.md](docs/design-spec.md); the JSON contracts
shared by the three layers are in [docs/CONTRACTS.md](docs/CONTRACTS.md); feature set v2
(frame rate / interpolation, speed buttons, pitch, fades and volume keyframes, stems on the
timeline, transitions, effects, installer) is specified in [docs/FEATURES_V2.md](docs/FEATURES_V2.md).

## Architecture

```
┌──────────────────────────── Tauri v2 window ────────────────────────────┐
│  React 19 + TypeScript (src/)                                            │
│   • Zustand document store (non-destructive Project, undo/redo)          │
│   • XState v5 workflow machine (idle → ingest → cuts → duplicates →      │
│     reframe → audio → assemble → ready)                                  │
│   • WebGL2 color pipeline (primary, WB, lift/gamma/gain/offset, HSL x8,   │
│     RGB curves, .cube 3D LUT, vignette/grain, masks)                     │
│   • Keyframe engine (cubic-bezier, bounce, elastic) + speed-ramp engine   │
│   • Canvas timeline (snapping, ripple, beat track, waveforms) + graph     │
│     drawer (keyframes & speed curves)                                    │
├────────────────────────── invoke / events ──────────────────────────────┤
│  Rust core (src-tauri/)                                                  │
│   • axum media server: HTTP-Range streaming, single-frame JPEG, thumbs   │
│   • ffmpeg/ffprobe bridge: probe, frames, thumbnails, waveform peaks     │
│   • keyframes crate: reference interpolators for export                  │
│   • pipeline runner: spawns Python, re-emits JSON-lines progress         │
│   • headless exporter: single ffmpeg graph, NVENC → libx264 fallback     │
├──────────────────────── child process (stdout JSON) ────────────────────┤
│  Python pipeline (pipeline/)                                             │
│   • shots: TransNetV2 (ONNX) → PySceneDetect fallback                    │
│   • perception: YOLO-World / Grounded SAM 2 + ByteTrack + CLIP/histogram │
│   • reframe: rule-of-thirds constrained crop solver, EMA / Savitzky-Golay│
│   • audio: loudnorm measurement, gain, beat detection                    │
│   • transitions: optical-flow continuity scoring                         │
│   • separate: Demucs v4 voice / background stems (Isolate voice)         │
│   • assemble: 2:30–3:00 timeline Project JSON                            │
└──────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites (Windows)

| tool | how |
|---|---|
| Node 20+ | https://nodejs.org |
| Rust stable (msvc) + VS Build Tools C++ | `winget install Rustlang.Rustup` |
| ffmpeg 6+ | `winget install Gyan.FFmpeg` (found automatically) |
| Python 3.11+ | `winget install Python.Python.3.12` |
| WebView2 | ships with Windows 11 |

## Run

```bash
npm install
cd pipeline && python -m venv .venv && .venv/Scripts/pip install -e ".[dev]" && cd ..
npm run tauri dev
```

`npm run dev` alone serves the UI in a browser with mocked native calls, useful
for layout work.

Optional ML extras (GPU, several GB): `pipeline/.venv/Scripts/pip install -r pipeline/requirements-ml.txt`
and drop weights into `pipeline/models/` (see `pipeline/README.md`). Without
them the pipeline runs in *lite* mode: PySceneDetect cuts, histogram-based
duplicate matching, OpenCV optical flow.

## Headless rendering

Render a saved project, or a pipeline analysis, without opening the app:

```bash
src-tauri/target/release/cappycat-cli export projects/cappycat-clips.analysis.json out.mp4 --preset h264_nvenc_mp4
```

Build it once with `cargo build --release --bin cappycat-cli` in `src-tauri/`. Presets are
`h264_nvenc_mp4` (default for .mp4), `h264_mp4` and `prores_mov` (default for .mov). Add
`--range <startMs> <endMs>` to render a section.

## Tests

```bash
npm test                              # vitest: engines + timeline edit semantics (src/state/edits.test.ts)
cd src-tauri && cargo test --workspace # Rust core + keyframes crate (uses real ffmpeg)
cd pipeline && .venv/Scripts/pytest    # solver, tracker, beats, shots, end-to-end analyze
```

In `npm run dev` (browser mode) a dev-only perf harness is on `window.__cappy`: open
`http://localhost:1420/?perf`, then `await __cappy.buildSynthetic()` (43 linked clips, 16 speed
ramps, keyframes, a LUT; `{ v2: true }` adds transitions on every 3rd cut, six effects, audio fades
and volume keyframes) and `await __cappy.measure(4)` / `measurePaused(2)` / `measureRedraw()`.
`await __cappyPreview.probe()` samples the program monitor (9 x 5 RGB grid) after the next frame.
`http://localhost:1420/?installed` makes the browser mock behave like a fresh install (the AI setup
wizard opens on first run, with simulated downloads).

## Workflow

1. **Import** raw clips and `.cube` LUTs into the media library (`Ctrl+I`, or drop files).
2. **AI Pipeline** — set open-vocabulary prompts (e.g. "raccoon in hoodie"), run.
   The inspector shows each stage; findings (duplicate collisions, rough cuts,
   loudness) appear when it finishes and the timeline is assembled and zoomed to fit.
   If another AI job (voice separation) holds the GPU, the run shows *waiting for the GPU* and
   starts by itself. *Rough cut in story order* lays the clips out without AI.
3. **Edit** (CapCut conventions) — the first video track is the **main track**: with the
   **Magnet** on (`P`) it stays gapless, deletes close the gap, trims / speed changes / freezes
   ripple, and dragging a clip inserts it between others. Video clips and their audio are
   **linked**: moves, trims, splits, deletes, speed, freeze, reverse and voice mode apply to both.
   Split with `Ctrl+B` (or `S`), trim edges (snapping to cuts, playhead and beats; `N` toggles),
   drag on an empty area to marquee-select, `Ctrl+A` selects all, Shift/Ctrl-click toggles.
   Speed presets (Montage, Hero Time, Bullet, Jump Cut, Flash In/Out) and custom curves, colour
   presets, wheels (drag the puck), HSL, curves, LUTs, masks, keyframes (`Alt+K`).
   `Ctrl+Alt+C` / `Ctrl+Alt+V` copy / paste attributes (grade, speed, motion, audio, mask);
   Color tab → *Apply to all clips* / *Apply to same source*. `C` toggles the raw-vs-graded compare
   view with detection boxes and the auto-zoom crop. Audio tab → **Voice separation**:
   *Original · Isolate voice · Remove vocals* per clip (or *Apply to all clips*); the stems are
   separated locally on the GPU (Demucs v4, `pipeline/models`, cached per file) and used by the
   preview and the export. Every drag or slider move is one undo step (`Ctrl+Z`, 200 steps).
   - **Speed:** quick buttons 0.25× … 5× (Speed tab, or right-click a clip → Speed) set a constant
     speed and ripple; *Optical flow* interpolates slow motion with RAFT at export. *Keep pitch*
     (Audio tab, on by default) time-stretches the voice instead of pitching it.
   - **Audio:** drag the top corners of an audio clip for equal-power fade in / out (video clips that
     carry their own sound: bottom corners), drag the volume line for the level, `Alt+click` it to add
     a volume keyframe (drag keyframes up / down, `Alt+click` one to remove it). Audio tab: volume
     (with a keyframe toggle at the playhead), fade in / out, keep pitch.
   - **Separate to tracks** (Audio tab or clip menu, *Apply to all clips*): runs the voice separation
     if needed and puts the **Voice** and **Background** stems on their own tracks as clips linked to
     the video clip (link groups can have any number of members); the original audio is muted, so it
     sounds the same, but each stem has its own volume, fades, keyframes and mute.
   - **Transitions** (left panel → Transitions): 16 types (dissolve, dips, wipes, slides, pushes,
     zooms, blur dissolve, flash, circle open) with animated previews. Drag one onto a cut, or click a
     cut's bow-tie marker and pick one; the Inspector sets the type and duration (0.1–3 s, centred on
     the cut, the timeline length does not change); `Delete` removes it; *Apply to all cuts*.
   - **Effects** (left panel → Effects): camera snap (flash, shutter sound, polaroid freeze), fades
     from / to black / white, black & white, sepia, letterbox, shake, zoom punch, blur in / out, RGB
     split, VHS, vignette pulse, flash. Click to add at the playhead or drag onto the FX track; trim
     and move like clips; the Inspector sets intensity and parameters. More FX tracks stack.
   - **Video fades:** Motion tab or the top corner handles of a video clip (from / to black).
   - **Right-click a clip:** split, delete, speed, separate to tracks, add transition, copy / paste
     attributes, freeze frame, reverse.
   - **Project settings** (File → Project settings…, or click the project chip): frame rate 24 / 30 /
     40 / 50 / 60 and frame interpolation (optical flow, frame blend, none), applied on export when
     the output rate is above a clip's effective rate; the Export dialog can override both.
4. **Save** — `Ctrl+S` / `Ctrl+Shift+S`; File → recent projects. Unsaved work is autosaved every
   60 s to `<cacheDir>/autosave/<projectId>.json` and offered for restore on the next start; New /
   Open / closing the window ask before discarding changes.
5. **Export** (`Ctrl+E`) — headless ffmpeg render with loudness normalisation; *Run in background*
   keeps the progress in the status bar. The frame rate and interpolation default to the project's.

**AI setup (installed app).** On the first run of the installed app, a wizard lists what is missing
(ffmpeg, the Python environment, the models), shows every download with its source URL and size,
and installs them on *Download and install* (progress per step, *Cancel*). *Skip* keeps the editor,
colour and export working; the AI buttons stay disabled with a hint until you run
**Help → AI setup…**. Help also has *Keyboard shortcuts* and *Open logs folder*. Running from the
repo (dev mode) never opens the wizard by itself.

### Keyboard shortcuts

Press `?` in the app for the full sheet.

| keys | action |
|---|---|
| `Space` | play / pause (restarts at the end) |
| `J` / `K` / `L` | shuttle reverse / pause / forward (press again for 2x, 4x) |
| `←` / `→` (`Shift`: 10 frames) | previous / next frame |
| `↑` / `↓` | previous / next cut |
| `Home` / `End` | start / end of the timeline |
| `Ctrl+B` or `S` | split at the playhead (selected clips, or every clip under it) |
| `Delete` | delete the selection, or the transition of the selected cut |
| `Alt+F` / `Alt+R` | freeze frame / reverse the selected clip |
| `Ctrl+Alt+C` / `Ctrl+Alt+V` | copy / paste attributes |
| `Ctrl+A` | select all clips |
| `Ctrl+Z` / `Ctrl+Y` (`Ctrl+Shift+Z`) | undo / redo |
| `N` / `P` | snapping / main-track magnet on/off |
| `Shift+Z`, `Ctrl+=` / `Ctrl+-` | zoom to fit, zoom in / out (`Ctrl+wheel` at the cursor) |
| `Alt+K` | keyframe / speed graph |
| `C` / `U` / `F` | compare view / universal adjust / large preview |
| `Ctrl+S` / `Ctrl+Shift+S` / `Ctrl+O` / `Ctrl+N` | save / save as / open / new |
| `Ctrl+I` / `Ctrl+E` | import / export |
| `Esc` | cancel a drag, close dialogs / menus / graph / large preview, clear the selection |
| right-click a clip | clip menu (split, delete, speed ▸, separate to tracks, add transition ▸, attributes, freeze, reverse) |
| click a cut's bow-tie | select the cut (Inspector → Transition) |
| drag a clip corner | fade in / out (video and audio) |
| `Alt+click` the volume line | add a volume keyframe (`Alt+click` a keyframe removes it) |
| `?` | shortcut cheat sheet |

Slider values: double-click the label to reset, click the number to type a value.

## Status

v0.1.0. The whole design is implemented and runs end to end on Windows: analysis with the full ML
stack (TransNetV2, YOLO-World / Grounded SAM 2, OpenCLIP cast identification, QP camera planner,
Demucs, RAFT), or in lite mode without it, and an exporter that matches the preview frame for frame
(colour, LUT, curves, masks, transforms, speed ramps, transitions, effects, 24-60 fps with optical
flow). There is also a per-user Windows installer with a first-run AI setup wizard.

See [docs/HANDOFF.md](docs/HANDOFF.md) for the current state, the decisions to keep, known limits and
next steps.

## License

Source code and documentation: [MIT](LICENSE). The character artwork in `characters/` is not
MIT-licensed; it is included for running and developing Cappycat only (see
[characters/LICENSE.md](characters/LICENSE.md)).
