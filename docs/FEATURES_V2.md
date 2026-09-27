# Cappycat feature set v2: shared specification

Speed and frame rate, pitch-accurate audio, stems on the timeline, audio fades and volume
keyframes, transitions, effects, and the Windows installer.

This is the contract the three layers implement. TypeScript (`src/types/project.ts`) is the source
of truth for field names. Rust (`src-tauri/src/model.rs`) and Python
(`pipeline/cappycat_pipeline/schema.py`) mirror it. Every new field is **optional**, and old
projects load with the defaults given here.

---

## 1. Frame rate and optical-flow frame interpolation

- `Project.fps` may be **24, 25, 30, 40, 48, 50 or 60** (plus 23.976 and 29.97 for sources).
  The UI offers 24 / 30 / 40 / 50 / 60 in *Project settings* and in the *Export* dialog.
- New `Project.frameInterpolation?: 'frameBlend' | 'opticalFlow' | 'none'`, default `'opticalFlow'`.
  - `none` repeats the nearest source frame.
  - `frameBlend` crossfades the two neighbouring source frames.
  - `opticalFlow` synthesises true in-between frames with RAFT, via the pipeline's `interpolate`
    command, whenever the output frame rate is higher than the clip's effective frame rate:
    `source avg fps × speed at that moment`. The exporter's existing slow-motion optical-flow path
    is generalised to cover this.
- Pipeline `interpolate` gains `--target-fps F`, an alternative to `--factor N`. It writes a video
  at exactly F fps covering `[in-ms, out-ms)`. Output frame k sits at source time `in + k/F`, built
  by flow-warping the bracketing source frames at the fractional position. When a timestamp falls
  exactly on a source frame, that frame is copied unchanged. Cuts are held, not morphed, as now.
  Results are cached (the exporter already caches by key; add the target fps to the key).
- The preview plays at the source frame rate. Interpolation is export-only, and the export dialog
  says so.

## 2. Speed changes

Speed ramps, presets and optical-flow slow-mo already exist. Add:
- Quick speed buttons in the Speed tab and the clip right-click menu: 0.25×, 0.5×, 0.75×, 1×,
  1.25×, 1.5×, 2×, 3×, 5×. Each sets a constant curve and ripples the timeline, as the store
  already does for speed.
- An "Optical flow" toggle next to it (`speed.opticalFlow`, already in the model), explained in a
  tooltip.

## 3. Pitch-accurate audio

- `ClipAudio.keepPitch?: boolean`, default **true** (CapCut "Pitch" off = keep pitch).
- **Export:** when speed ≠ 1 and `keepPitch` is set, time-stretch the clip's audio along the clip's
  speed map with a **pitch-preserving algorithm**. Use WSOLA or a phase vocoder, implemented in
  Rust, that handles variable speed (ramps) natively.
  - Tempo range 0.1× to 10×; quality good on speech (the clips are dialogue).
  - `keepPitch = false` keeps today's varispeed resampling (pitch follows speed), with an
    anti-alias low-pass when speed > 1.
  - Reverse and freeze behave as now.
- **Preview:** `HTMLMediaElement.preservesPitch = keepPitch` on every media element that plays the
  clip's audio.

## 4. Audio fades and volume keyframes

- `ClipAudio.fadeInMs?: number` and `ClipAudio.fadeOutMs?: number`, default 0, measured in
  timeline ms and clamped to half the clip length each. The curve is equal-power (sine), as in
  CapCut.
- `ClipAudio.volume?: Keyframed<number>`: a dB offset added to `gainDb`, keyframed in clip-local
  timeline ms with the same `Keyframed` type and easing semantics as transforms. Its static value
  is 0.
- Final gain at clip-local time t is
  `dbToLin(gainDb + volume(t)) × fadeIn(t) × fadeOut(t)`, or 0 when muted.
- **UI:**
  - Fade handles at the top-left and top-right corners of every audio clip (and of video clips
    that carry audio), drawn as a ramp over the waveform, as in CapCut.
  - A volume line across the clip. Clicking the line with Alt adds a volume keyframe, and
    keyframes are dragged up and down.
  - Inspector Audio tab: Volume (dB), Fade in (s), Fade out (s), Keep pitch, and a keyframe
    toggle for volume.

## 5. Voice stems as editable timeline clips

- **Action "Separate to tracks"** in the Audio tab and the clip right-click menu:
  1. Runs voice separation for the clip's asset if its stems are missing (`separate_audio`,
     which already exists).
  2. Registers two **audio assets**, one per stem: `kind: 'audio'`, path = the stem file,
     `durationMs` = the source's duration, name `"<clip> · Voice"` / `"<clip> · Background"`, and a
     new `Asset.stemOf?: { assetId: string; stem: 'vocals' | 'background' }`.
  3. Creates the audio tracks **"Voice"** and **"Background"** if missing, with
     `Track.role?: 'voice' | 'background'`.
  4. Adds one clip per stem, aligned with the source clip: same `startMs`, `inMs`, `outMs`,
     `speed`, `reversed` and `freezeFrame`, and the same `linkId` group as the video clip.
  5. **Mutes** the original linked audio clip (`audio.muted = true`), so the result sounds the
     same as before, but each stem now has its own volume, fades, keyframes, mute, voice mode
     (ignored for stem clips) and waveform.
- **Linked groups can have more than 2 members.** Every edit that follows links applies to the
  whole group.
- "Apply to all clips" also runs *Separate to tracks* for every audio-bearing clip.
- **Export:** stem clips are plain audio clips of audio assets, which works today. The rule that
  de-duplicates mirrored audio must not drop them, because their asset differs from the video's
  asset.

## 6. Transitions (between two clips on the same video track)

- `Clip.transitionIn?: { type: TransitionType; durationMs: number } | null` on the **incoming**
  clip. It describes the transition from the previous clip on the same track that ends exactly
  where this clip starts, with a gap under 1 frame.
- The transition is **centred on the cut**, from `cut − d/2` to `cut + d/2`. The timeline length
  does **not** change.
  - The outgoing clip keeps playing past its out-point using source frames beyond `outMs` when the
    source has them (handles). Otherwise it holds its last frame.
  - The incoming clip likewise starts early using frames before `inMs`, or holds its first frame.
  - Speed maps extend linearly at the boundary speed.
- `durationMs` ranges from 100 to 3000 (default 500) and is clamped to the shorter of the two
  clips.
- `TransitionType` and its meaning, with p = progress 0 → 1 across the window, A = outgoing clip,
  B = incoming clip:

| type | meaning |
|---|---|
| `dissolve` | crossfade: A·(1−p) + B·p |
| `dipToBlack` | fade A to black over the first half, black up to B over the second (colour `#000`) |
| `dipToWhite` | same through white |
| `wipeLeft` / `wipeRight` / `wipeUp` / `wipeDown` | hard-edged wipe with a 2% soft edge, B revealed in the given direction |
| `slideLeft` / `slideRight` | B slides in over a static A |
| `pushLeft` / `pushRight` | B pushes A out |
| `zoomIn` | A scales 1 → 1.3 while fading out; B scales 0.8 → 1 while fading in |
| `zoomOut` | the reverse scales |
| `blurDissolve` | crossfade while both clips blur up to 12 px at the midpoint |
| `flash` | crossfade through a white flash peaking at the midpoint |
| `circleOpen` | B revealed inside a growing circle from the centre, with a soft edge |

- Easing: progress runs through an ease-in-out (the `easeInOut` cubic-bezier) for every type.
- **UI:**
  - The left panel gets tabs **Media · Transitions · Effects · Audio FX**. Audio FX is optional
    and can be omitted.
  - The Transitions tab shows animated thumbnails of each type, rendered live with the WebGL
    renderer or as simple CSS previews.
  - Drag a transition onto a cut, or select a cut and click a transition, to apply it. A small
    bow-tie marker appears on the cut; click it to change the type or duration, press Delete to
    remove it.
  - "Apply to all cuts" sets the chosen transition on every cut of the main track.
  - The preview renders transitions live.

## 7. Effects (clips on the FX track)

- An effect is a clip on an `fx` track with `assetId: ''` and
  `effect: { type: EffectType; intensity: number /* 0..1, default 1 */; params?: Record<string, number> }`.
  The clip's `startMs` and its duration (`outMs − inMs`, speed ignored) define when it applies.
- An effect applies to the **composited frame** of all video tracks during its span, in track
  order; multiple FX tracks stack.
- Every effect's strength follows an envelope: a 120 ms ramp in and a 120 ms ramp out, times
  `intensity`. Types that define their own timing ignore the envelope.
- `EffectType` and its meaning (t = seconds since the effect started, D = the effect's duration):

| type | meaning |
|---|---|
| `cameraSnap` | "Take a photo". A white flash (alpha 1 → 0 over 250 ms) at t = 0. The underlying picture **freezes on its t = 0 frame** for D. During the freeze the frame shrinks to 92% with a 3% white border and a soft drop shadow, over a 15% darkened, blurred copy of the same frame as background (a polaroid look). A shutter sound plays at t = 0: a procedurally synthesised click about 120 ms long, mixed at −6 dB. Params: `border` (0..0.1, default 0.03), `scale` (0.8..1, default 0.92). Default duration 1500 ms. |
| `fadeFromBlack` | Black → picture over D (linear in light). No envelope. |
| `fadeToBlack` | Picture → black over D. No envelope. |
| `fadeFromWhite` / `fadeToWhite` | As above, through white. |
| `blackAndWhite` | Desaturate (luma), with a slight contrast boost of 1.1×. |
| `sepia` | Classic sepia matrix. |
| `letterbox` | Cinematic bars; param `ratio` (default 2.39). Bars slide in and out with the envelope. |
| `shake` | Camera shake. Translation and rotation follow deterministic value noise; params `amplitude` (px as a fraction of width, default 0.01) and `frequency` (Hz, default 12). |
| `zoomPunch` | Quick push-in: scale 1 → 1.15 → 1 over D with an ease-out-back curve. |
| `blurIn` / `blurOut` | Gaussian blur 20 px → 0, or 0 → 20 px, over D. No envelope. |
| `rgbSplit` | Glitch: R and B channels offset horizontally by `amount` (default 0.006 × width), jittering with value noise at 8 Hz. |
| `vhs` | Scanlines, slight chroma bleed, noise and tracking-line wobble. |
| `vignettePulse` | Vignette strength oscillating 0.2 ↔ 0.6 at 1 Hz. |
| `flashWhite` | White flash that peaks at D/2. |

- The Effects tab lists them with previews. Drag one onto the FX track, or click to add it at the
  playhead with its default duration. The clip can be trimmed and moved like any other clip, and
  the Inspector shows intensity and the type's params.
- The preview renders every effect live. The **exporter must match the preview**, using the same
  maths as the GLSL: Rust in `src-tauri/src/render/`.
- The shutter sound comes from `procedural_shutter(sample_rate) -> Vec<f32>` in Rust, with a
  matching WebAudio buffer in the preview, built the same way so both sound alike. It is a short
  high-passed noise burst plus two clicks.

## 8. Video fades on clips

`Clip.fadeInMs?` and `Clip.fadeOutMs?` (video), default 0. They fade the clip's picture from or to
**black** over its first or last ms, using clip-local timeline time. This is equivalent to
`fadeFromBlack` / `fadeToBlack` limited to one clip. The Inspector Motion tab gets two sliders, and
the timeline draws corner handles on video clips, as for audio.

## 9. Windows installer (`npx tauri build` → NSIS installer)

- **App data layout** when running installed (not from the repo):
  - `%LOCALAPPDATA%\Cappycat\`: `python\` (the managed virtual env), `models\`, `ffmpeg\`,
    `cache\`, `logs\`, `autosave\`.
  - `Documents\Cappycat\`: `Clips\`, `Projects\`, `Exports\`, `Characters\`, `Presets\`.
  - On first run, `Characters` and `Presets` are seeded from the bundled defaults.
  - The pipeline source is bundled as a Tauri resource (`pipeline/cappycat_pipeline/**`,
    `pipeline/requirements*.txt`, `pipeline/pyproject.toml`).
- **Dev mode is unchanged:** repo-relative, as today. The Rust core resolves every location through
  one `paths` module:
  1. an env override (`CAPPYCAT_HOME`, `CAPPYCAT_PIPELINE_DIR`, …);
  2. the repo layout, if found;
  3. the installed layout.
- **First-run AI setup** is a wizard in the UI, and can be re-run from *Help → AI setup*. Command
  `setup_ai { components?: string[] }` → job id, with events `setup://progress`
  `{ jobId, step, pct, message }` and `setup://done { jobId, ok, error }`. Steps:
  1. **ffmpeg:** use one on PATH or in WinGet if present. Otherwise download the official
     static build zip from `https://github.com/BtbN/FFmpeg-Builds/releases` (latest
     `ffmpeg-master-latest-win64-gpl.zip`; show the URL and size, and ask the user to confirm in
     the wizard) and extract it to `ffmpeg\`.
  2. **Python environment** via `uv`:
     - Download `uv.exe` from its official GitHub release (`astral-sh/uv`,
       `uv-x86_64-pc-windows-msvc.zip`) into `%LOCALAPPDATA%\Cappycat\bin\`, or use one on PATH.
     - `uv venv --python 3.12 <python dir>` (uv downloads a managed CPython).
     - `uv pip install` torch/torchvision from the cu130 index (**CPU wheels when no NVIDIA GPU is
       detected**), then `pipeline/requirements-installer.txt`, then `pip install --no-deps` for
       the packages the pipeline already installs that way (demucs etc.).
  3. **Models:** `python -m cappycat_pipeline download-models` with progress.
  4. **Verify:** `python -m cappycat_pipeline doctor`; the wizard shows the result.

  Every download is shown with its source URL before it starts, and the user can skip AI setup;
  the editor, colour and export features work without it. Python 3.14 is used in dev; the
  installer uses 3.12 for wheel availability, so `requirements-installer.txt` pins versions known
  to work on 3.12.
- **Bundle:** the NSIS installer is per-user (no admin), with an optional desktop shortcut and a
  Start-menu entry. MSI is optional and can be skipped if WiX is unavailable. The installer does
  not include Python, models or ffmpeg (the wizard installs them), so it stays small. Its size is
  reported.
- **Uninstall** removes the app. It asks whether to also remove `%LOCALAPPDATA%\Cappycat`, which
  holds the models and Python, and it never touches `Documents\Cappycat`.
