# Cappycat pipeline

Local computer-vision / audio analysis for the Cappycat editor. The Rust core runs it as

```
python -m cappycat_pipeline analyze <clip...> --out analysis.json [options]
python -m cappycat_pipeline interpolate <src> --in-ms A --out-ms B --factor N --out <file.mp4>
python -m cappycat_pipeline interpolate <src> --in-ms A --out-ms B --target-fps F --out <file.mp4>
python -m cappycat_pipeline separate <media...> --json
python -m cappycat_pipeline download-models          # installer / first run
python -m cappycat_pipeline doctor --json            # installer verification
```

and reads one JSON event per line from stdout (`progress`, `log`, `result`; see
`docs/CONTRACTS.md`). Everything else goes to stderr: the process's fd 1 is redirected to stderr
and events are written to a private duplicate, so even native libraries (CUDA, onnxruntime) cannot
corrupt the stream.

Stages per clip: **ingest → shots → perception → reframe → audio → transitions**, then the
cuts between clips are scored and a global **assemble** writes an `AnalysisResult` (analysis per
clip + assembled `Project` timeline) to `--out` (atomically: `<out>.<pid>-<id>.part` + `os.replace`).
Before the clip loop, missing models are downloaded (`download` progress events) and every clip is
looked up in the per-clip result cache; a fully cached run loads no model at all.

## Install (base / "lite" mode)

Python 3.10+ (3.14 tested). From `pipeline/`:

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"      # or: pip install -r requirements.txt
.venv\Scripts\python -m cappycat_pipeline doctor
```

Base dependencies: `numpy`, `opencv-python-headless`, `scenedetect`, `onnxruntime`, `scipy`.
ffmpeg 6+ must be installed (`winget install Gyan.FFmpeg`). Binaries are located via
`CAPPYCAT_FFMPEG_DIR` → `PATH` → the WinGet package folder.

## Install (ML path, GPU)

```powershell
# 1. torch FIRST, from the PyTorch index for your driver (cu130 tested; cu126/cu128 for older drivers)
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 2. the rest (never pulls a CPU torch; see the comments in the file)
.venv\Scripts\pip uninstall -y onnxruntime          # replaced by onnxruntime-gpu (same module name)
.venv\Scripts\pip install -r requirements-ml.txt    # or: pip install -e ".[ml]" (+ ".[onnx-gpu]")
# 3. every model into pipeline/models (idempotent; JSON progress on stdout)
.venv\Scripts\python -m cappycat_pipeline download-models
.venv\Scripts\python -m cappycat_pipeline doctor      # -> "mode": "ml", "modelsReady": true, device, free VRAM
```

Tested set: torch 2.14.0+cu130, torchvision 0.29.0+cu130, ultralytics 8.4.162 (+ the
`ultralytics/CLIP` prompt encoder), open_clip_torch 3.3.0, transformers 5.17.0, accelerate 1.15.0,
huggingface_hub 1.33.0, librosa 1.0.0, onnx 1.23.0, onnxscript 0.7.2, onnxruntime-gpu 1.30.0
(CUDA 13 build; it finds CUDA/cuDNN through the torch install), demucs 4.1.0 (+ julius 0.2.8,
einops 0.8.2, lameenc 1.8.4, sphn 0.2.1). demucs 4.1 needs `torchaudio` only for training and the
audio is decoded with ffmpeg here, so torchaudio is not installed (there is no torchaudio build for
torch 2.14 / cp314 on the cu130 index); if pip ever wants to re-resolve torch, install it with
`pip install --no-deps demucs==4.1.0 julius einops lameenc sphn`. Grounding DINO and SAM 2 come from
Hugging Face `transformers`, so there is no CUDA extension to compile. ultralytics is run with
`YOLO_AUTOINSTALL=False` / `YOLO_OFFLINE=True` and a private settings dir, so it never pip-installs
anything (which could swap the CUDA torch) and sends no telemetry.

### Models (`pipeline/models/`, or `CAPPYCAT_MODELS_DIR`) — all git-ignored

| file / cache | size | used by |
|---|---|---|
| `transnetv2.pt` | 29 MB | shot detection, torch (official TransNetV2 weights, PyTorch port `Sn4kehead/TransNetV2`) |
| `transnetv2.onnx` | 29 MB | the same network exported by `download-models` (onnxruntime path, checked against torch) |
| `yolov8s-worldv2.pt` | 25 MB | YOLO-World detector |
| `clip/ViT-B-32.pt` | 338 MB | YOLO-World prompt encoder (only used while `set_classes` runs, on the CPU) |
| `hf/hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K` | 577 MB | OpenCLIP duplicate / identity embeddings |
| `hf/hub/models--IDEA-Research--grounding-dino-tiny` | 658 MB | Grounded SAM 2 boxes |
| `hf/hub/models--facebook--sam2.1-hiera-small` | 176 MB | Grounded SAM 2 masks |
| `torch/hub/checkpoints/raft_small_C_T_V2-*.pth` | 4 MB | RAFT optical flow |
| `hf/hub/models--adefossez--HTDemucs-ft` | 321 MB | voice / background separation (Demucs v4 `htdemucs_ft`, MIT; 4 fine-tuned nets) |

`HF_HOME`, `TORCH_HOME` and `YOLO_CONFIG_DIR` default into this folder (set them yourself to
override). The TransNetV2 network definition is vendored (MIT) in `cappycat_pipeline/transnetv2_model.py`;
the three public PyTorch conversions on the hub are bit-identical and reproduce the reference behaviour
(hard cut p≈0.99 on the boundary frame, dissolve peak ≈0.97 mid-fade, static/pan ≈0).

## What runs where

| stage | lite (no torch) | ML path |
|---|---|---|
| shots | PySceneDetect `AdaptiveDetector` | TransNetV2: **torch CUDA** > onnxruntime (CUDA/DirectML/CPU) > torch CPU > PySceneDetect |
| perception | skipped | YOLO-World v2-s (**CUDA fp16**) + ByteTrack; Grounded SAM 2 (**CUDA**, Grounding DINO fp16, SAM 2.1 fp32) on escalation; OpenCLIP ViT-B-32 (**CUDA fp16**) embeddings |
| characters | skipped | OpenCLIP identity against the main-cast references (`<repo>/characters`) |
| reframe | numpy | numpy |
| audio | ffmpeg `ebur128` + numpy beats | librosa beat tracker (numpy periodicity gate + tempo agreement) |
| transitions | OpenCV DIS flow | RAFT-small (**CUDA fp32**) |
| interpolate | DIS (or Farneback) flow + CPU warping | RAFT flow + **GPU** warping/blending (`grid_sample`) |
| separate | – (needs demucs + torch) | Demucs v4 `htdemucs_ft` (**CUDA fp16 autocast**, CPU fallback) |

`CAPPYCAT_DEVICE=cpu` forces CPU; `CAPPYCAT_FLOW_BACKEND=auto|raft|dis|farneback` picks the flow backend.

### VRAM

Budget: `CAPPYCAT_VRAM_BUDGET_MB` (default 2048, capped by `torch.cuda.mem_get_info` free − 256 MB).
Models load lazily and are released (`del` + `torch.cuda.empty_cache()`) after each stage; YOLO-World's
CLIP text encoder is dropped after the prompts are encoded and OpenCLIP keeps only its image tower.
Measured peaks (`torch.cuda.max_memory_allocated`, RTX 4070 Laptop): TransNetV2 ~150 MB,
YOLO-World + OpenCLIP ~0.6 GB, with a Grounded SAM 2 escalation ~1.0–1.9 GB (Grounding DINO fp16,
up to 4 captions per batched forward; SAM 2.1 fp16, only in the perception pass - duplicate tracking
runs without masks). A CUDA OOM (or any other error) during an escalation keeps the shot's
YOLO-World result, frees the fallback, logs a `warn` and disables escalation for the rest of the run. RAFT's all-pairs
correlation volume grows with (H·W)², so flow runs at ≤ 720p with resolution / batch chosen from the
budget (1280×720 source: flow at 1036×583, ~1.7 GB peak for one flow).

## Duplicate characters

Per sampled frame (`--sample-fps`, default 4) of every shot:

1. Detect with the prompts (class-agnostic NMS at IoU 0.6, so a character matched by several prompts
   is one instance), track with a class-agnostic ByteTrack whose thresholds follow the detector's own
   confidence (tracks only provide stable ids; pairs are tested per frame from the detections).
2. Embed every instance with OpenCLIP; tall boxes also by their top 45 % ("head"). **Group boxes**
   (> 55 % of the frame area, or full height and > 70 % of the width - one box around several
   characters) get no identity and no cast tag.
3. **Named rule** (main cast loaded): two instances identified as the *same* unique character whose
   mutual similarity is ≥ `--similarity-threshold` → finding with `character` / `characterName`. Two
   unconfirmed instances with the same best-guess character and near-identical appearance count too.
4. **Generic rule** (unnamed extras / no manifest): label-compatible, both ≥ 15 % of the frame tall,
   similar scale, similarity ≥ threshold; with the cast loaded and no explicit `--prompts`, both must
   also look cast-like (best cast score ≥ id threshold − 0.12), which keeps birds, paws and props out.
5. Overlapping (IoU > 0.3) or nested boxes are never duplicates; the primary stays on the same side
   for the whole shot so the reframe crop doesn't alternate.
6. A duplicate (per character / track pair) is only reported when it was seen in **≥ 2 sampled
   frames** - one misidentified pair must not trigger dense tracking and a reframe.

**Following a confirmed duplicate through its shot** (`dupetrack.py`, `--track-fps 8`, 0 = off):
once a pair is confirmed, the shot is re-scanned at 8 fps (streamed: only detections / embeddings
are kept, shared by every duplicate group of the shot; frames the perception pass already detected
come from a detection cache verified by a pixel checksum) and both the primary and the duplicate are
followed forwards and backwards from the confirmed frames by box continuity (IoU / centre distance to
a constant-velocity prediction) plus appearance similarity to their embeddings at the confirmed frame
they started from / last passed. The
identity score is not needed again, but a detection confidently identified as a *different* cast
member (full box and head crop, like the perception pass) is never taken. Missed instances are held
for 1 s (the duplicate with a +5 % margin); going forwards the duplicate then counts as gone, going
**backwards** it keeps being excluded at its earliest known position all the way to the shot start
unless that position touches a frame edge (it walked in). The reframe then plans the camera over the
whole shot (see "Camera path"); tracking runs without SAM masks and reports per-frame progress. Each
tracked shot logs `tracked duplicate <name> at 8 fps: {...}`. When tracking fails, the sampled-frame
reframe is used; it holds the crop 0.25 s beyond the first / last finding and eases back to the full
frame over 0.5 s (it used to zoom for the whole shot).

`--detector hybrid` (default) runs YOLO-World + ByteTrack and **escalates the whole shot** to
Grounded SAM 2 (sampled at `--escalation-fps`, default 2; dense tracking fills in) when YOLO-World
finds nothing for the prompts, when two label-compatible instances that **may be the same character**
overlap (IoU > 0.3; not when their best identities differ and both score ≥ 0.66, not for a head box
inside its body box, not when either is shorter than 15 % of the frame), or when a pair's similarity is borderline
(`threshold − 0.1 … threshold`). Shots where YOLO-World finds fewer than 1.5 detections per frame are
re-run once at 1280 px (full-resolution frames, `--no-hires` to skip): small background figures are
missed at 640. Each shot logs its path, e.g. `clip7.mp4 shot 4: path=grounded_sam2 (escalated:
possibly-same instances overlap (IoU 0.52 > 0.3; ...)); cast [Bunny]; … 1 duplicate(s): Bunny x2 @ 13.10s`.

## Main cast (`<repo>/characters/characters.json`)

Loaded automatically when present (`--characters PATH`, `--no-characters`). Reference crops are
cut from the character sheets (Grounding DINO boxes; boxes shorter than 25 % of the sheet must pass an
OpenCLIP zero-shot "character, not an accessory" check - full figures skip it, because a back view is
mostly backpack for CLIP) and from the group lineup (faces sorted left → right, assigned in the listed
order), embedded with flipped and head variants, and cached in `characters/.cache/refs.npz` (keyed by
the sha1 of the manifest + images, written atomically; the crops go to `characters/.cache/crops/` for
review, `*_ref.jpg` = reference only, `*_dropped.jpg` = near-duplicate). A reference entry may list
manual crop boxes (`"boxes": [[x1, y1, x2, y2], ...]`, image pixels; detection is then skipped unless
`"detect": true`). Near-identical reference crops of one character (cosine > 0.97, e.g. a sheet's
"front" and "neutral" views) are dropped, and a crop's flipped / head variants count once: identity =
mean of the top-3 per-crop similarities per character, assigned when ≥ `idThreshold` and ahead of the
runner-up by `idMargin`. Both are calibrated leave-one-image-out (lineup crops vs sheets only, sheet
crops vs the lineup only; the queries are the zero-shot-accepted crops, as before), then clamped to
0.68–0.74 (threshold) and 0.05–0.08 (margin; 0.02 used to let a group box through as "Suzie").

**Open-set rejection.** The bank also stores OpenCLIP text embeddings: per character its species
("a cartoon rabbit character") and a description (`"zeroShot": [...]` in the manifest, default
"a cartoon <species> character: <features>"), plus the manifest's `negatives` (`[{"text": "a cartoon
deer"}, {"image": "x.webp", "boxes": [[...]]}]`) and every `notLike` text. A detection whose zero-shot
mass on the negatives is ≥ 0.5 and above its best character's (or that is closer to a negative image
reference than to its best character) gets no identity - e.g. the elderly turtle is not Turtle, while
the young Turtle stays Turtle because its description wins.

**Cast** (`Shot.cast`): ids confidently identified in ≥ 2 sampled frames, plus ids carried along a
ByteTrack track by a looser rule (score ≥ 0.66 and margin ≥ 0.05 - or a zero-shot vote ≥ 0.8 for the
same character instead of the margin; score ≥ 0.62 when the zero-shot vote is ≥ 0.95, e.g. a back
view - in ≥ 3 frames and the majority of the track's frames).
`Asset.sceneTags` = the names in the clip. Detection itself uses generic prompts (`genericPrompts` +
"person", "cartoon character"): on the real clips they found far more of the characters than
per-species prompts.

## CLI

```
python -m cappycat_pipeline analyze <clips...> --out <json>
    [--prompts "raccoon in hoodie" "person"]
    [--detector yolo_world|grounded_sam2|hybrid|none]     # default hybrid (degrades to yolo_world -> none)
    [--shot-detector transnetv2|pyscenedetect|auto] [--shot-threshold 0.5]
    [--characters PATH | --no-characters]
    [--target-duration-ms N] [--target-lufs -14] [--similarity-threshold 0.85]
    [--smoothing ema|savgol] [--smoothing-alpha 0.15] [--smoothing-window 15]
    [--no-normalize-audio] [--no-beats] [--sample-fps 4] [--escalation-fps 2] [--analysis-width 640]
    [--track-fps 8] [--no-hires] [--no-cache] [--no-download] [--keep-order]
    [--director] [--ollama-url http://127.0.0.1:11434] [--ollama-model qwen2.5-vl]
python -m cappycat_pipeline interpolate <src> --in-ms A --out-ms B (--factor N | --target-fps F) --out <file.mp4>
python -m cappycat_pipeline order <folder | files...> [--recursive] [--json]
python -m cappycat_pipeline separate <media...> [--model htdemucs_ft|htdemucs] [--device cuda|cpu] [--force] [--json]
python -m cappycat_pipeline download-models [--only NAME...]
python -m cappycat_pipeline characters build [--if-stale] | identify <image|video> [--at-ms N] [--detector ...]
python -m cappycat_pipeline probe <path> | shots <path> [--shot-detector ...] | beats <path> | doctor [--json]
python -m cappycat_pipeline --watch-stdin <command> ...     # (or CAPPYCAT_WATCH_STDIN=1) see "Cancellation"
```

### Result cache, first-run downloads, progress

* **Per-clip result cache** (`resultcache.py`): every successfully analysed clip is stored under
  `%LOCALAPPDATA%\cappycat\cache\analysis\clips\<sha1>.json` (`CAPPYCAT_ANALYSIS_CACHE_DIR` overrides),
  keyed by the clip's absolute path, size and mtime, every option that changes its analysis, the
  pipeline / analysis version, the character-bank hash (manifest + reference images) and which models
  are installed. Cross-clip transitions are cached per clip pair (`pairs/`). A hit emits the stage
  progress events with message `"cached"` and a log line `clipN.mp4: cached analysis reused (...)`.
  `--no-cache` neither reads nor writes. Results of a run that degraded (a model missing, an
  escalation OOM, a failed stage) are not cached.
* **Downloads**: before the clip loop the models the run needs (for its detector / shot detector /
  a stale character bank) are fetched when missing, with `"stage": "download"` progress events
  (`--no-download` skips). When everything is local, `HF_HUB_OFFLINE=1` is set, so offline runs never
  wait on the network (`_hf_load` also skips the online retry then).
* **Progress**: 2 % set-up, 95 % clips (weighted by duration; a cached clip ≈ 3 % of an analysed one),
  3 % cross-clip transitions + assembly. Per clip: ingest 1 %, shots 6 %, perception 55 %, reframe
  (duplicate tracking + camera planning) 30 %, audio 4 %, transitions 4 %. Perception and tracking
  report per frame (`shot 3/5: Grounded SAM 2 frame 7/10`, `shot 5: tracking frame 21/47`).

### Cancellation and atomic outputs

Every output is written to `<name>.<pid>-<id>.part` next to the target and moved over it with
`os.replace` only when complete: the analysis JSON, `interpolate`'s mp4 (`...part.mp4`), `separate`'s
stems (temporary folder, then replace), the character cache (`refs.npz`) and the per-clip result
cache. Every ffmpeg / ffprobe goes through `procs.popen` / `procs.run`. With `--watch-stdin` (anywhere
on the command line) or `CAPPYCAT_WATCH_STDIN=1`, a daemon thread reads stdin; at EOF (the parent
died or closed the pipe to cancel) it kills every live child process and exits with code 3. It is
opt-in so running the CLI from a terminal is unaffected; the Rust core keeps stdin open as a pipe and
sets the variable (it also uses a Windows Job Object).

### Audio

Loudness is measured with ffmpeg `ebur128=peak=true` (within 0.6 LU of `loudnorm`, ~3.7x faster;
`loudnorm` is the fallback). The recommended gain is limited by the true peak:
`gain = min(target − I, −1 − TP)`, clamped to ±12 dB (clip8 used to get +9.6 dB → +5.5 dBTP). Beats
are only emitted when the audio has a pulse: numpy onset-autocorrelation periodicity ≥ 0.25 (dialogue
measured 0.03–0.16) **and** librosa's tempo agrees with numpy's (ratio ≈ 1, 2 or ½). The per-clip
`audio.beatConfidence` (0–1, optional field) says how much the grid is trusted; the log line shows
periodicity, both tempos and the reason when beats are skipped. The loudness is per clip (every shot
of the clip uses the clip's value, which is also the right fallback for short shots whose integrated
loudness would be unreliable).

### Timeline assembly

A shot's video clip and its mirrored audio clip share a `linkId` (`Clip.linkId`, optional, omitted
when None). When the total exceeds `--target-duration-ms`, the longest shots are trimmed at **both
ends** (the middle is kept; with a confident beat grid the out point is moved onto a source beat
within the shot); if even the 1.5 s floors don't fit, the **lowest-value** shots (fewest cast members,
shortest, duplicate-artifact shots first; ties drop the later one) are dropped and the rest re-fitted.

**`interpolate`** (used by the exporter for optical-flow slow motion): decodes `[A, B)` at the source
frame rate and writes `fps × N` video (libx264, crf 16, yuv420p, no audio) with every original frame
followed by `N − 1` synthesised frames: RAFT forward + backward flow at ≤ 720p (upsampled),
Super-SloMo intermediate flows, both neighbours warped and blended with forward/backward-consistency
occlusion weights. The in-betweens after the last frame interpolate towards the first frame after
`B` (held when the source ends), so the output has exactly `K × N` frames — the segment's duration.
Pairs across a hard cut are held instead of morphed. Variable-frame-rate sources (AI generators
often store 24 fps on a 1/60 timebase) are resampled onto their average rate first. Frames stream in
VRAM-bounded batches. stdout: `{"event":"progress","stage":"export","clip":"<name>","pct":…,"message":…}`
lines, then `{"event":"result","path":"<out>"}`; exit 0 (1 with an error `log` event on failure).
Throughput on a 1280×720 VFR clip, ×4: ~20 s for a 2 s segment (RTX 4070 Laptop, incl. start-up).

**`interpolate --target-fps F`** (frame-rate conversion for export, FEATURES_V2 §1; exclusive with
`--factor`) writes `[A, B)` at exactly `F` fps: `round((B − A) · F)` frames, encoded with the exact
rational rate (`60` → 60/1, `29.97` → 30000/1001, `23.976` → 24000/1001; `30000/1001` is accepted too).
Output frame `k` is the source at `A + k/F`:

* the source is decoded once with its **real frame timestamps** (`-fps_mode passthrough` + a
  `showinfo` filter in the same ffmpeg, `-copyts`, pts × time base − container start time), so VFR
  sources (24 fps content on a 1/60 or 1/15360 timebase) need no resampling;
* a source frame within 0.5 ms of `A + k/F` is **copied unchanged** (byte-identical to the decoded
  frame); otherwise the bracketing frames `t_i ≤ T < t_{i+1}` are RAFT flow-warped at
  `α = (T − t_i)/(t_{i+1} − t_i)` (the same bidirectional warp + forward/backward-consistency
  occlusion blend as `--factor`, at an arbitrary α); pairs across a hard cut (same detector as
  `--factor`) are **held** (the outgoing frame before α = 0.5, then the incoming one); output times
  before the first / after the last decodable frame hold that frame;
* windows of `batch + 1` source frames are processed together: on CUDA the frames are uploaded once,
  RAFT runs on area-downscaled copies (≤ 720p, forward + backward flows batched as the VRAM budget
  allows, `plan_flow`), flows are upsampled and every in-between is warped on the GPU, and only the
  finished frames come back (`transitions.TorchFlowInterpolator`). Without CUDA (or after a GPU
  error / OOM) the CPU path runs: DIS flow + OpenCV warping;
* same events as `--factor` (`export` progress, then the result), plus one `info` log line with
  frame counts (copied / interpolated / held / cut-held), backend, time and peak VRAM; atomic
  output, the stdin watchdog kills decoder and encoder.

Measured on `cappycat- clips/clip1.mp4` (1280×720, 24 fps content on a 1/15360 VFR timebase),
0–5 s → 60 fps: 300 frames (120 copied, 180 synthesised) in 24.5 s, i.e. ~12.2 output frames/s
(~4.9 source frames/s), peak VRAM 1203 MB allocated (RTX 4070 Laptop; RAFT at 1036×583 dominates:
~100 ms per flow, two flows per source pair, batch 1 within the 2 GB budget).

**`separate`** (CapCut "Isolate voice" / "Remove vocals", `separate.py`) splits each file's audio
into a `vocals` stem (dialogue) and a `background` stem (drums + bass + other: ambience, music, SFX)
with Meta's Demucs v4 (`htdemucs_ft` by default: the 4 fine-tuned nets of the bag, vocals from the
vocals specialist; `--model htdemucs` is the single net, ~4× faster, slightly worse vocals). Audio is
decoded with ffmpeg (44.1 kHz stereo for the model), processed in the model's 7.8 s segments with 25 %
overlap while the full mix stays on the CPU (VRAM stays bounded whatever the length), fp16 autocast
on CUDA, CPU when CUDA is missing / the free VRAM is below ~1.1 GB / the GPU runs out of memory.
Stems are written as 48 kHz stereo float WAVs to
`%LOCALAPPDATA%\cappycat\cache\stems\<sha1(path|size|mtime|model)>\{vocals,background}.wav` (+ `meta.json`;
`CAPPYCAT_STEMS_DIR` overrides) and reused when present. They are sample-aligned with the source and
share its timeline: an audio stream that starts after the container's start is padded with silence,
and they end where the source audio ends (exactly what the exporter decodes). With `--json` stdout
carries `{"event":"progress","stage":"separate","clip":name,"pct":overall,"clipPct":file,"message":..}`,
`log` events (one `info` per file with timing / peak VRAM / stem RMS, `error` for a failed file) and
one `{"event":"result","path":<source as given>,"stems":{"vocals":..,"background":..}}` per file;
exit code 1 when any file failed. Measured on `cappycat- clips/clip1.mp4` (15 s, RTX 4070 Laptop):
~12 s per call (torch import + 1.4 s model load + ~4.5 s inference + resampling / writing), peak VRAM
534 MB allocated / 758 MB reserved; a cached file answers in ~0.6 s. The very first GPU run on a machine
can take minutes while CUDA builds its kernel cache.

**`order`** orders the media of a folder (or a list of files) by what their names say: markers
(`ep1 scene2 shot3`, `S01E02`, `part II`), leading / trailing numbers, ordinal words, timestamps,
intro/outro words — with a reason per file and warnings for gaps or ties. `analyze` uses the same
ordering for folders unless `--keep-order`.

**`download-models`** fetches everything above (idempotent: present models are reported as
`"<name>: already present"` and not fetched); progress events use `"stage": "download"` with the
model name as `clip`, then `{"event":"result","path":"<models dir>"}`. Each progress event also
carries byte counts: `bytes` / `totalBytes` for the current model and `overallBytes` /
`overallTotalBytes` for the whole run (only what is missing counts). They are updated about twice a
second while a model downloads, from the growth of its download folder (HF `blobs/*.incomplete`,
torch.hub's temporary file, ...); `totalBytes` is the model's known size (`downloads.EXPECTED_BYTES`)
and is raised to `bytes` if a download turns out larger. `pct` is the overall fraction weighted by
those sizes. Exit code 1 when a model failed (an `error` log event names it).

**`doctor [--json]`** prints one JSON object (the output is always JSON; library versions via
`importlib.metadata`, so only torch is imported): ffmpeg, library versions, `device`, `vram`
(`torch.cuda.mem_get_info`), `vramBudgetMb`, each model's presence / size / location,
`modelsReady`, `shotBackend`, `flowBackend`, the character manifest / cache state and `mode`, plus
for the installer's AI-setup wizard: `packages` (every required package → version or null),
`cudaAvailable`, `nvidiaGpu` (`{name, driver}` from `nvidia-smi`, or null), `missing`
(`"ffmpeg"`, `"package:<name>"`, `"model:<name>"`), `warnings` (e.g. an NVIDIA GPU with a CPU-only
torch) and `ready` (= nothing missing; CUDA is not required).

### Installer requirements (Python 3.12)

The installed app runs the pipeline on a uv-managed CPython **3.12** (dev uses 3.14):

1. `torch==2.14.0` + `torchvision==0.29.0` from `https://download.pytorch.org/whl/cu130` (NVIDIA
   GPU) or `.../whl/cpu`; cp312 win_amd64 wheels of both exist on both indexes (checked with
   `pip index versions` and the index pages).
2. `requirements-installer.txt`: the whole dependency closure pinned (the dev-tested versions),
   binary wheels only, torch / torchvision excluded; CPU `onnxruntime`.
3. `requirements-installer-nodeps.txt` with `--no-deps`: demucs 4.1.0, julius, einops, lameenc,
   sphn and ultralytics' CLIP fork (GitHub source archive of the tested commit: pure Python, no git
   needed).

Verified without installing: `pip download --only-binary=:all: --python-version 3.12 --platform
win_amd64 --implementation cp` of `requirements-installer.txt` with `--no-deps` (87 wheels) and
with dependencies (`torch==2.14.0` / `torchvision==0.29.0` as constraints: exactly the pinned set +
torch + torchvision, so the pins are complete and consistent), the no-deps wheels likewise, and the
CLIP archive builds a wheel.

The assembled project's fps snaps to the nearest standard rate (23.976, 24, 25, 29.97, 30, 48, 50,
59.94, 60) when within 0.5 %: the AI clips' average rate of 24.04 becomes a 24 fps project.

Transitions: `smoothness` blends flow and colour continuity; hard cuts are normal, so `suggestion`
is `"dissolve"` only below 0.12 or when a cut is < 0.2 and under half of the clip's median.

`--director` sends each shot's middle frame + detections to a local Ollama VLM using the
"Automated Cinematic Director" prompt; it is skipped silently when Ollama is unreachable.

## Tests

```powershell
.venv\Scripts\python -m pytest -q          # everything; GPU tests skip without CUDA / models / media
.venv\Scripts\python -m pytest -q -m gpu   # only the ML-path tests
```

`tests/test_review_fixes.py` and `tests/test_camera_fallbacks.py` cover the behaviours added by the
pipeline review: the escalation rule (different cast members / head-in-body / possibly-same), the OOM
fallback (a fake Grounded SAM 2 raising `torch.cuda.OutOfMemoryError`), ≥ 2-frame duplicate evidence,
group boxes, track-level cast tagging, open-set rejection, reference dedupe, the detection cache, the
hi-res re-run, true-peak-limited gain, `ebur128` parsing, beat gating / tempo agreement, atomic
writes (incl. a cancelled `interpolate`), non-ASCII image paths, the stdin watchdog (a child process
is killed when stdin closes), the per-clip result cache (hit / miss / `--no-cache` / mtime) and
monotonic progress, side-flip segment splitting, soft exclusion, the QP frame cap, secondary primaries,
the sampled-frame reframe relaxing at the ends and holding an unseen duplicate back to the shot start;
`tests/test_gpu.py::test_grounding_dino_batched_prompts_match_one_forward_per_prompt` checks that the
batched Grounding DINO forward gives the same boxes as one forward per prompt (and is faster). The
camera-feel tests in `tests/test_camerapath.py` are unchanged. `tests/test_target_fps.py` covers
`--target-fps` on a synthetic textured square moving 10 px per frame at 24 fps: 24 → 60 and 24 → 30
(frame counts, exact rates incl. 30000/1001, copies byte-identical to the decoded source, in-betweens
with the square within 3 px of its linearly interpolated position, on the CPU (DIS) and with RAFT on
CUDA), a hard cut held not morphed, a VFR source copied at its real timestamps, argument errors and a
cancelled run leaving nothing behind. `tests/test_setup.py` covers `doctor --json` (`ready` / `missing`)
and the byte progress / idempotence of `download-models`; `tests/test_schema.py` the FEATURES_V2
fields (round trip, omitted when absent, enums checked against `src/types/project.ts`). Current
counts: 171 passed + 1 skipped (`-m "not gpu"`), 23 passed (`-m gpu`).


The base suite generates synthetic clips with ffmpeg (a 6 s `testsrc` → red → blue video, and the same
on a 1/60 VFR timebase) and runs `interpolate` with the DIS fallback on the CPU. `tests/test_gpu.py`
checks TransNetV2 (torch vs ONNX, cuts / dissolves), YOLO-World fp16, Grounded SAM 2 masks, OpenCLIP,
RAFT, `doctor`/`download-models`, an end-to-end duplicate-person clip built from ultralytics'
`bus.jpg` (hybrid and grounded_sam2), the character calibration and, when `<repo>/cappycat- clips`
is present, the clip7 duplicate-Bunny / clip9 two-turtles cases. `tests/test_separate.py` covers the
stem cache, WAV I/O, the audio-offset convention and the `separate` CLI contract without a model; its
`gpu` tests run Demucs (both models) on a synthetic mix of a formant-synthesised voice (glottal pulses
with vibrato through vowel formants, syllables and pauses) over a drum loop + noise bed and check that
the vocals stem follows the voice (corr > 0.5, bed < 0.15), the background stem the bed (corr > 0.9),
equal lengths, the cache hit, and that stems of a clip whose audio starts 0.5 s late stay on the
source's timeline.

## Layout

```
cappycat_pipeline/
  schema.py            dataclasses mirroring src/types/project.ts + to_json() (incl. the optional
                       FEATURES_V2 fields: frameInterpolation, keepPitch, audio fades / volume
                       keyframes, stemOf, track role, transitionIn, effect, video fades; None =
                       absent = the spec default in DEFAULTS_V2, omitted from the JSON)
  ffmpeg_util.py       ffmpeg/ffprobe discovery, probe, rawvideo frame pipe (VFR-safe), PCM decode
  models.py            model registry, cache dirs (HF/torch/ultralytics), device + VRAM helpers
  downloads.py         download-models steps (+ TransNetV2 ONNX export)
  transnetv2_model.py  vendored TransNetV2 network (MIT)
  shots.py             TransNetV2 (torch / onnxruntime) / PySceneDetect AdaptiveDetector
  tracking.py          ByteTrack (Kalman + two-stage IoU association), numpy
  perception.py        YOLO-World / Grounded SAM 2 / hybrid detectors, OpenCLIP / histogram embedders,
                       per-frame duplicate finding, hybrid escalation
  dupetrack.py         follows confirmed duplicate pairs through their shot (streamed dense re-scan, shared)
  characters.py        main-cast manifest, reference bank, calibration, identify
  reframe.py           rule-of-thirds crop solver, EMA / Savitzky-Golay smoothing, ReframeTrack builder
  camerapath.py        whole-shot camera path QP (Clarabel / OSQP), side splitting, soft fallback, QP frame cap
  audio.py             ebur128 loudness (loudnorm fallback), true-peak-limited gain, gated beat tracking
  transitions.py       RAFT / DIS flow, transition scoring, occlusion-aware interpolation (GPU / CPU)
  interpolate.py       the `interpolate` subcommand (streaming, VRAM-bounded)
  separate.py          Demucs v4 voice / background stems (the `separate` subcommand), stem cache
  assemble.py          ClipAnalysis[] -> Project timeline (linked clips, trimming, shot dropping)
  resultcache.py       per-clip analysis result cache (+ cross-clip transition pairs)
  fsutil.py            atomic writes (.part + os.replace), non-ASCII-safe image I/O
  procs.py             child-process registry + the stdin (parent-death) watchdog
  ordering.py          clip order from filenames
  director.py          Ollama AI-director prompt + client (urllib)
  cli.py               argparse entry point, JSON-lines events
```

## Camera path (smooth reframing)

When a duplicate is tracked through a shot, the reframe camera is planned for the whole shot at
once (`camerapath.py`). The duplicate is a hard constraint and never enters the crop. Keeping
the main character's head in frame is a strong soft rule. The path otherwise follows the ideal
per-frame framing while penalising movement (L1, so the camera holds still),
acceleration and jerk (L2, so moves ease in and out with no jitter). The camera starts
easing in before a duplicate walks into frame instead of snapping when it appears. Tune the feel with
`CameraStyle` (`follow`, `hold`, `accel`, `jerk`). The solver is Clarabel, with OSQP as the fallback;
the old smooth-then-clamp path is still available with `build_tracked_reframe(..., camera="smooth")`.

When one side per visibility segment is infeasible (a duplicate crosses the primary while it stays
visible), the plan is retried with the segments split where the geometric side flips (0.25 s
hysteresis), then with the exclusion as a heavily weighted soft constraint (the per-frame clamp still
enforces it); feasible shots get exactly the first plan. With several duplicates in one shot, the
other pairs' primaries are kept in frame by a lighter soft head term. Shots longer than 1440 frames
(60 s at 24 fps) are planned on an evenly decimated grid with the objective rescaled to per-frame
units and interpolated back, which bounds the QP size (and the adaptive re-plans).
