# Cappycat — handoff

State of the project as of 2026-09-27 (v0.1.0). Read this first, then `README.md` (setup, run,
tests), `docs/design-spec.md` (the original design), `docs/CONTRACTS.md` (JSON shared by the three
layers) and `docs/FEATURES_V2.md` (feature set v2).

## What it is

A local, offline AI video editor for short AI-generated clips. You drop a folder of clips in; the
pipeline orders them by filename, finds the camera cuts, spots duplicated characters (a generator
artefact: the same character twice in one frame), reframes them out with a smooth virtual camera,
levels the audio and lays everything out on a CapCut-style timeline. The timeline stays fully
editable. It exports H.264 MP4 or ProRes MOV at 24 to 60 fps.

## What works (verified)

- **Analysis** of the 10-clip sample project (`cappycat- clips/`, 15 s each, 1280×720, 24 fps content
  on a 60 fps timebase) takes about 146 s from cold on an RTX-class GPU; a per-clip result cache
  makes unchanged clips instant. It produces 43 shots, identifies the six-character main cast
  against `characters/`, and finds the two duplicate-character shots and reframes them.
- **Cut timing** comes from the real frame timestamps (`ffmpeg_util.FrameClock`). Cut points are
  safe for the preview and for every export frame rate (`assemble.cut_points`).
- **Editor:** timeline, undo/redo, inspector, colour (WebGL2), transitions (16) and effects (16)
  with live preview, speed and keep-pitch, stems (voice / background) as linked audio clips, fades
  and volume keyframes, and the universal colour preset. Playback runs at 60 fps.
- **Exporter** (Rust, CPU compositor): matches the preview frame for frame (colour, LUT, curves,
  masks, transforms, speed ramps, transitions, effects). It uses NVENC with a libx264 fallback and
  can reach 30 to 60 fps with RAFT optical flow. Headless: `cappycat-cli export`.
- **Windows installer:** NSIS (per-user) plus MSI. On first run a wizard installs uv, Python 3.12,
  CUDA torch and the model weights.
- **Tests:** all pass: pipeline 176 (non-GPU) + 23 (GPU), Rust workspace 178, vitest 93.

## Repo map

| Path | Layer |
|---|---|
| `src/` | React 19 + TS UI: `state/` (Zustand store, edits, persistence), `engine/` (colour, keyframes, speed, audio, transitions, effects), `components/` |
| `src-tauri/src/` | Rust core: `media_server.rs` (token-guarded HTTP), `export/` (decode, compositor feed, audio, stretch, flow, encode), `render/` (colour, LUT, masks, transitions, fx, time map), `setup.rs` (AI setup), `bin/cappycat-cli.rs` |
| `pipeline/cappycat_pipeline/` | Python ML: `shots`, `perception` (detection and duplicates), `dupetrack`, `reframe` + `camerapath` (QP camera planner), `audio`, `separate` (Demucs), `interpolate` (RAFT), `assemble`, `cli` |
| `characters/` | Main-cast reference sheets + `characters.json` (names, prompts, negatives). Fox and Felix are the same character. |
| `presets/universal-adjust.json` | House colour look, on by default, toggle in the UI |
| `cappycat- clips/`, `projects/`, `exports/`, `installers/` | Local media and outputs, git-ignored |

## Decisions to keep

- **Leave the "AI" label** in the source clips; do not build watermark removal.
- **Camera overshoot:** the reframe camera pushes in slightly past the target and settles. This is
  wanted, and the planner (`camerapath.py`, `max_overshoot` 0.06) adapts the amount per shot.
- **Universal preset** uses CapCut's scale: adjust sliders ±50, HSL ±100 (hue ±30°). Values are in
  `presets/universal-adjust.json`.
- **UI palette:** black, grey and blue. No orange.
- **Preview and export parity:** the GLSL in `src/engine/color/*` and the Rust in `src-tauri/src/render/*`
  implement the same maths. Change both together; each side's tests pin the shared constants.
- **VFR sources:** never convert frame index to time with `index / avg_fps`. Use
  `ffmpeg_util.frame_clock(path)`. Decode with `-fps_mode passthrough`.

## Build and release

```bash
npm install && npm run build                      # UI
cd src-tauri && cargo build --release             # app + cappycat-cli
npx tauri build                                   # NSIS + MSI (set CARGO_TARGET_DIR to an ABSOLUTE path if you move it)
```

Installers land in `src-tauri/target/release/bundle/{nsis,msi}/`. The last built ones are copied
to `installers/` (git-ignored). The installer bundles the pipeline source, `characters/` and
`presets/`; the ML runtime is installed on first run.

## Known limits and next steps

- **No captions or subtitles:** no speech-to-text model yet. Whisper would slot in next to `separate`.
- **16:9 only:** a vertical (9:16) edit needs a second reframe pass targeting the new aspect. The
  crop solver and camera planner already take the aspect as a parameter.
- **Duplicate removal is reframe-only** (crop the duplicate out). Inpainting is not implemented.
- **Shot trimming** (when the edit exceeds the target length) centres the window or snaps it to a
  beat. It does not yet score frames for content.
- **Windows-first:** paths, the Job Object child-process handling and the installer are Windows. The
  core should port, but it is untested on macOS and Linux.
- **One GPU job at a time**, by design, so analysis and export queue rather than compete for VRAM.

## Deliverables of the sample project

`exports/repurpose-kit/` (see its README) holds the master MP4, all 43 shots as separate clips,
stills with a contact sheet, the lossless mix, dialogue and background stems, and a shot list
(CSV/JSON) with cast and timecodes.
