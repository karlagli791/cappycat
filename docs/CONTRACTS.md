# Cappycat shared contracts

All three layers (React frontend, Rust/Tauri core, Python pipeline) exchange data
using the JSON shapes below. The TypeScript source of truth is `src/types/project.ts`;
Python mirrors it in `pipeline/cappycat_pipeline/schema.py`; Rust mirrors it in
`src-tauri/src/model.rs`. All times are **milliseconds** (integers or floats),
all pixel coordinates are in **source pixel space** of the asset, bounding boxes are
`[x1, y1, x2, y2]`.

## Project (timeline document)

```jsonc
{
  "version": 1,
  "id": "proj_...",
  "name": "Ep_01",
  "fps": 24,
  "width": 1920,
  "height": 1080,
  "assets": [Asset],
  "tracks": [Track],
  "beatMarkers": [ { "timeMs": 1234.5, "strength": 0.8, "kind": "beat1" | "beat2" } ]
}
```

### Asset
```jsonc
{ "id": "ast_...", "path": "C:/abs/path.mp4", "name": "Clip_01.mp4",
  "kind": "video" | "audio" | "image" | "lut",
  "durationMs": 50000, "width": 1920, "height": 1080, "fps": 24, "hasAudio": true,
  "codec": "h264", "sceneTags": ["Bunny","Turtle"], "order": 0, "orderReason": "clip 1",
  "stems": { "vocals": "C:/.../cache/stems/<sha1>/vocals.wav",          // optional: set once the file
             "background": "C:/.../cache/stems/<sha1>/background.wav" } } //   was separated (see below)
```

### Track
```jsonc
{ "id": "trk_...", "kind": "video" | "audio" | "fx", "name": "Video 1",
  "locked": false, "muted": false, "clips": [Clip] }
```

### Clip
```jsonc
{
  "id": "clp_...", "assetId": "ast_...", "trackId": "trk_...",
  "startMs": 0,            // position on the timeline
  "inMs": 0, "outMs": 5000, // source range (before speed)
  "speed": SpeedCurve,
  "transform": { "position": Keyframed<[x,y]>, "scale": Keyframed<number>,
                 "rotation": Keyframed<number>, "opacity": Keyframed<number>, "blur": Keyframed<number> },
  "color": ColorGrade,
  "audio": { "gainDb": 0, "normalize": true, "muted": false,
             "voice": "original" | "voice" | "background" },   // default "original" (see Voice separation)
  "mask": null | { "shape": "rectangle"|"circle"|"split"|"filmstrip", "feather": 0.1,
                   "rect": Keyframed<[x,y,w,h]>, "inverted": false },
  "blendMode": "normal"|"multiply"|"screen"|"overlay"|"softLight"|"darken"|"lighten"|"colorDodge",
  "reframe": null | ReframeTrack,
  "label": "Shot 2", "freezeFrame": null | { "atMs": 1200, "holdMs": 800 }, "reversed": false
}
```

### Keyframed<T>
```jsonc
{ "static": T, "keyframes": [ { "timeMs": 0, "value": T,
      "easing": "linear"|"easeIn"|"easeOut"|"easeInOut"|"bounce"|"elastic"|"bezier",
      "bezier": [x1, y1, x2, y2] } ] }
```
`timeMs` is relative to the clip's start on the timeline. If `keyframes` is empty, `static` is used.

### SpeedCurve
```jsonc
{ "preset": "normal"|"montage"|"hero_time"|"bullet"|"jump_cut"|"flash_in"|"flash_out"|"custom",
  "points": [ { "t": 0.0, "speed": 1.0 }, ... ],  // t in [0,1] over the SOURCE range, speed in [0.1, 10]
  "opticalFlow": true }
```
Speed between points is interpolated with a monotone cubic; playback time is the integral of 1/speed.

### ColorGrade
```jsonc
{ "exposure": 0, "brilliance": 0, "contrast": 0, "brightness": 0, "highlights": 0, "shadows": 0,
  "saturation": 0, "vibrance": 0, "sharpness": 0, "temperature": 0, "tint": 0,
  "lift": [0,0,0], "gamma": [0,0,0], "gain": [0,0,0], "offset": [0,0,0],
  "hsl": { "red": {"h":0,"s":0,"l":0}, "orange": ..., "yellow": ..., "green": ...,
           "cyan": ..., "blue": ..., "purple": ..., "magenta": ... },
  "curves": { "master": [[0,0],[1,1]], "r": [[0,0],[1,1]], "g": [[0,0],[1,1]], "b": [[0,0],[1,1]] },
  "lutAssetId": null, "lutIntensity": 1.0, "vignette": 0, "grain": 0 }
```
All scalar sliders are in `[-100, 100]` (0 = neutral) except `lutIntensity`, `vignette`, `grain` in `[0, 1]`... see `src/engine/color/defaults.ts`.

### ReframeTrack (output of the auto-zoom solver)
```jsonc
{ "sourceWidth": 1920, "sourceHeight": 1080,
  "keyframes": [ { "frame": 0, "timeMs": 0, "crop": [x1,y1,x2,y2], "zoom": 1.45, "tx": -0.12, "ty": 0.0 } ],
  "reason": "duplicate raccoon excluded on right boundary" }
```
`zoom` is the scale factor to apply to the source, `tx`/`ty` are normalised translations in [-1, 1] of the source frame.

## Pipeline analysis result (`cappycat-pipeline analyze`)

```jsonc
{
  "version": 1,
  "generatedAt": "2026-09-24T20:00:00Z",
  "clips": [
    {
      "path": "...", "asset": Asset,
      "shots": [ { "index": 0, "startFrame": 0, "endFrame": 119, "startMs": 0, "endMs": 4958.3,
                   "confidence": 0.93, "method": "transnetv2" | "pyscenedetect",
                   "cast": ["bunny", "turtle"] } ],
      "duplicates": [ { "shotIndex": 0, "frame": 40, "timeMs": 1666,
                        "primary": { "trackId": 1, "label": "raccoon in hoodie", "bbox": [..], "score": 0.9 },
                        "duplicate": { "trackId": 2, "label": "raccoon in hoodie", "bbox": [..], "score": 0.8 },
                        "similarity": 0.94, "character": "bunny", "characterName": "Bunny" } ],
      "reframe": [ { "shotIndex": 0, "track": ReframeTrack } ],
      "audio": { "integratedLufs": -23.1, "truePeakDb": -1.2, "recommendedGainDb": 9.0, "beats": [ms...], "tempoBpm": 120 },
      "transitions": [ { "fromShot": 0, "toShot": 1, "flowMagnitude": 3.2, "smoothness": 0.71,
                         "suggestion": "cut" | "dissolve" } ]
    }
  ],
  "timeline": Project   // assembled 2:30–3:00 sequence
}
```

## Pipeline progress events (stdout, one JSON per line)

```jsonc
{ "event": "progress", "stage": "ingest"|"shots"|"perception"|"reframe"|"audio"|"transitions"|"assemble"|"export"|"download"|"separate",
  "clip": "Clip_01.mp4" | null, "pct": 0.0-1.0, "message": "human readable" }
{ "event": "log", "level": "info"|"warn"|"error", "message": "..." }
{ "event": "result", "path": "C:/.../analysis.json" }
```
Rust re-emits these as Tauri events named `pipeline://progress`, `pipeline://log`, `pipeline://result`.

## Rust ↔ frontend commands (Tauri `invoke`)

| command | args | returns |
|---|---|---|
| `probe_media` | `{ path }` | `Asset` |
| `import_media` | `{ paths: string[] }` | `Asset[]` |
| `media_server_url` | — | `"http://127.0.0.1:PORT"` |
| `extract_thumbnails` | `{ path, count, width }` | `string[]` (URLs on media server) |
| `extract_waveform` | `{ path, samplesPerSecond }` | `number[]` peaks in [0,1] |
| `run_pipeline` | `{ paths: string[] (files or folders), options: PipelineOptions (incl. keepOrder) }` | job id; `pipeline://progress|log|result|analysis|exit` |
| `cancel_pipeline` | `{ jobId }` | — |
| `separate_audio` | `{ paths: string[] }` (media files) | job id; `separate://progress` `{ jobId, pct, clipPct, message, clip }`, `separate://result` `{ jobId, path, stems }`, `separate://log` `{ jobId, level, message }`, `separate://done` `{ jobId, ok, error }` (`error: "cancelled"` after cancel) |
| `cancel_separate` | `{ jobId }` | — |
| `export_project` | `{ project: Project, outPath, preset?, range?: { startMs, endMs } }` (`h264_mp4`, `h264_nvenc_mp4`, `prores_mov`, or `<preset>_legacy`) | job id; `export://progress` `{ jobId, pct, message }`, `export://log` `{ jobId, level, message }`, `export://done` `{ jobId, ok, outPath, error }` |
| `cancel_export` | `{ jobId }` | — |
| `scan_clips_folder` | `{ path? }` (default: `<repo>/clips` if it has videos, else the newest `<repo>/*clip*` folder with videos) | `{ folder, assets: Asset[] (with order + orderReason), warnings: string[] }` |
| `save_project` / `load_project` | `{ path, project? }` | `Project` |
| `evaluate_keyframes` | `{ keyframed, timeMs }` | value (Rust interpolator crate; outgoing-keyframe easing) |
| `app_paths` | — | `{ cacheDir, thumbsDir, analysisDir, ffmpeg, ffprobe, python, pipelineDir, clipsDir, mediaServerUrl }` |

Media server routes:
- `GET /media?path=<abs>` — byte-range file streaming (HTML5 `<video>`/`<audio>` source)
- `GET /frame?path=<abs>&ms=<n>&w=<n>` — single JPEG frame via ffmpeg
- `GET /thumb/<hash>.jpg` — cached thumbnails

## Voice separation (CapCut "Isolate voice" / "Remove vocals")

`ClipAudio.voice` picks what a clip plays: `original` (default; unknown values load as `original`),
`voice` (the asset's `stems.vocals`: dialogue only) or `background` (`stems.background`: ambience,
music, SFX without the voices). The exporter decodes that stem through the clip's in/out range,
speed map, reverse, freeze and gain; without stems it logs a `warn` on `export://log` and uses the
original audio. A video clip mirrored by an audio-track clip (same `assetId` / `startMs`) is heard
through the mirror, so the mirror's `voice` applies; the UI sets both together.

Stems come from `python -m cappycat_pipeline separate <media...> [--model htdemucs_ft|htdemucs]
[--device cuda|cpu] [--force] --json` (Meta Demucs v4, `htdemucs_ft` by default; `background` = drums +
bass + other). stdout, one JSON object per line:

```jsonc
{ "event": "progress", "stage": "separate", "clip": "clip1.mp4", "pct": 0.43,   // pct: over all files
  "clipPct": 0.86, "message": "clip1.mp4: model 4/4, segment 11/12" }           // clipPct: this file
{ "event": "log", "level": "info" | "warn" | "error", "message": "..." }        // error = a file failed
{ "event": "result", "path": "<source as given>", "stems": { "vocals": "...", "background": "..." } }
```

Stems are 48 kHz stereo float WAVs cached under `%LOCALAPPDATA%\cappycat\cache\stems\<sha1(path|size|mtime|model)>\`
(`CAPPYCAT_STEMS_DIR` overrides; a hit returns immediately). They share the source's timeline: sample
0 is the file's time 0 (an audio stream starting later than the container is padded with silence) and
they end where the source audio ends, so `-ss X` on a stem and on the source give the same instant and
a `<audio>` element can use the `<video>` element's `currentTime`. Exit code 1 when any file failed.

## Clip ordering

Clips are ordered by the sequence in their filenames (`python -m cappycat_pipeline order <folder> --json`
→ `{ folder, files: [{ path, name, order, reason, key }], warnings }`): scene/shot/part/clip markers,
S01E02 codes, leading numbers, ordinal and number words, trailing numbers, timestamps, then
intro/outro words; numbers compare numerically (clip2 < clip10). See `clips/README.md`.

## Main cast

`characters/characters.json` lists the unique main characters with reference images. The pipeline
identifies detections against them; the same unique character twice in one frame is a
`DuplicateFinding` with `character`/`characterName`. `Shot.cast` lists the character ids in a shot and
`Asset.sceneTags` the character names in a clip.

## Universal adjust

`Project.universalAdjust?: { enabled, name, values }` is layered on top of every clip's grade in the
preview and the export. `values` holds optional slider deltas (`exposure`, `brilliance`, `contrast`,
`brightness`, `highlights`, `shadows`, `saturation`, `vibrance`, `sharpness`, `temperature`, `tint`,
`vignette`, `grain`) and an optional per-channel `hsl` table. Each value is added to the clip's own
value and clamped to the slider range. The shared preset lives in `presets/universal-adjust.json`
(`{ version, name, enabledByDefault, values }`). New projects and headless renders of projects that
don't carry their own setting use it. Commands: `load_universal_adjust` returns `{ path, preset }`,
and `save_universal_adjust { preset }` returns the same. Adjust sliders use CapCut's scale: −50..50,
with sharpness 0..50. HSL offsets are −100..100, and HSL hue uses the CapCut scale: ±100 = ±30°.

## Linked clips

`Clip.linkId?: string` is shared by a video clip and its mirrored audio clip. The pipeline sets it per
shot (`lnk_…`), and the UI sets it on manual adds and rough cuts. Older projects get links
inferred on load: same asset, start and source range on a video and an audio track. Every edit
(move, trim, split, delete, speed, freeze, reverse, voice, gain) applies to both clips.
`audio.beatConfidence` (0–1, optional) is reported per clip; beats are skipped when unconfident.

## Security and new commands

- `media_server_url` returns `http://127.0.0.1:<port>/t/<token>` (a random per-launch token).
  All routes live under that prefix. Only the app's origins may call it cross-origin, and only
  registered media paths are served. `register_media { paths }` registers extra paths.
  Import, scan, `open_document`, analysis results, separation stems and export outputs register
  their paths automatically.
- `load_lut { path }` returns `{ size, data (RGB floats, red fastest), domainMin, domainMax, title? }`.
- `open_document { path }` returns `{ kind: 'project' | 'analysis', project, analysis? }`.
- The webview no longer has file-system access; the `fs` and `shell` plugins are removed.
- One GPU job runs at a time (analysis, voice separation, optical-flow interpolation). A queued job
  reports progress `pct: 0` with "waiting for the GPU (another AI job is running)".
- Exports render to `<out>.partial-<jobId>.<ext>` and are renamed over the target only on success.
  An output path equal to any source clip, LUT or stem is refused.
- Python children get `CAPPYCAT_WATCH_STDIN=1` and a piped stdin that the app holds open. When stdin
  closes, the pipeline exits and kills its own children. All children also live in a
  kill-on-close Windows Job Object.

## Feature set v2

Frame rate and optical-flow interpolation, pitch-accurate audio, audio fades and volume keyframes,
voice stems as timeline clips, transitions, effects, video fades, and the installer with its
AI setup. See [FEATURES_V2.md](FEATURES_V2.md) for the fields, formulas and commands.
The exact transition and effect constants live in `src/engine/color/fxShaders.ts` and are
mirrored 1:1 in `src-tauri/src/render/{transitions,fx}.rs`.
