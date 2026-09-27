//! Serde mirror of `src/types/project.ts` (see `docs/CONTRACTS.md`).
//!
//! Every struct uses camelCase field names and `#[serde(default)]` so partial
//! documents load (missing fields take neutral defaults). Times are in
//! milliseconds, geometry in source pixels, bounding boxes `[x1, y1, x2, y2]`.
//!
//! `Keyframed`, `Keyframe`, `Easing`, `SpeedCurve`, `SpeedPoint` and
//! `SpeedPreset` are re-exported from the `keyframes` crate so the evaluator
//! and the document share one definition.

use serde::{Deserialize, Serialize};

pub use keyframes::{Easing, Keyframe, Keyframed, SpeedCurve, SpeedPoint, SpeedPreset};

pub type Vec2 = [f64; 2];
pub type Vec3 = [f64; 3];
/// x, y, w, h
pub type Rect = [f64; 4];
/// x1, y1, x2, y2
pub type BBox = [f64; 4];
/// x, y in 0..1
pub type CurvePoint = [f64; 2];

/// A string enum that never fails to deserialise: known names map to their variant, anything
/// else becomes `Other(name)` (logged at `warn`) and serialises back unchanged, so a document
/// written by a newer frontend survives a load / save round trip through Rust. Every consumer
/// treats `Other` as the documented safe fallback of its type.
macro_rules! string_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident = $s:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant,)+
            /// A name this build does not know (kept verbatim).
            Other(String),
        }

        impl $name {
            /// Every known variant (not `Other`).
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            pub fn as_str(&self) -> &str {
                match self {
                    $($name::$variant => $s,)+
                    $name::Other(s) => s.as_str(),
                }
            }

            pub fn parse(s: &str) -> Self {
                match s {
                    $($s => $name::$variant,)+
                    other => $name::Other(other.to_string()),
                }
            }

            pub fn is_known(&self) -> bool {
                !matches!(self, $name::Other(_))
            }
        }

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                // any JSON value is accepted: a non-string becomes Other("<json>")
                let v = serde_json::Value::deserialize(d)?;
                let s = match &v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let e = $name::parse(&s);
                if !e.is_known() {
                    tracing::warn!("unknown {} '{}' in the document; using its safe fallback", stringify!($name), s);
                }
                Ok(e)
            }
        }
    };
}

/* ---------------------------------------------------------------- assets */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetKind {
    #[default]
    Video,
    Audio,
    Image,
    Lut,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Asset {
    pub id: String,
    pub path: String,
    pub name: String,
    pub kind: AssetKind,
    pub duration_ms: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub has_audio: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<u32>,
    /// Why the pipeline placed the asset at `order` (e.g. "leading number 01").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_reason: Option<String>,
    /// Voice / background stems (`separate_audio`), set once the file has been separated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stems: Option<Stems>,
    /// Set on the audio assets "Separate to tracks" registers for a stem file: the source asset
    /// and which stem this is. For the exporter these are plain audio assets; the mirrored-audio
    /// de-duplication never drops their clips.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stem_of: Option<StemOf>,
}

string_enum! {
    /// `Asset.stemOf.stem`
    StemKind { Vocals = "vocals", Background = "background" }
}

/// `Asset.stemOf`: `{ assetId, stem: 'vocals' | 'background' }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StemOf {
    #[serde(default)]
    pub asset_id: String,
    pub stem: StemKind,
}

/// Separated audio of an asset: 48 kHz stereo WAVs on the source's timeline (`docs/CONTRACTS.md`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Stems {
    pub vocals: String,
    pub background: String,
}

/* ---------------------------------------------------------------- tracks */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackKind {
    #[default]
    Video,
    Audio,
    Fx,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Track {
    pub id: String,
    pub kind: TrackKind,
    pub name: String,
    pub locked: bool,
    pub muted: bool,
    pub clips: Vec<Clip>,
    /// `'voice' | 'background'` on the audio tracks "Separate to tracks" creates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<TrackRole>,
}

string_enum! {
    /// `Track.role`
    TrackRole { Voice = "voice", Background = "background" }
}

/* ------------------------------------------------------------- transform */

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClipTransform {
    /// normalised offset of the clip centre, [-1,1]
    pub position: Keyframed<Vec2>,
    /// 1 = fit
    pub scale: Keyframed<f64>,
    /// degrees
    pub rotation: Keyframed<f64>,
    /// 0..1
    pub opacity: Keyframed<f64>,
    /// px
    pub blur: Keyframed<f64>,
}

impl Default for ClipTransform {
    fn default() -> Self {
        Self {
            position: Keyframed::constant([0.0, 0.0]),
            scale: Keyframed::constant(1.0),
            rotation: Keyframed::constant(0.0),
            opacity: Keyframed::constant(1.0),
            blur: Keyframed::constant(0.0),
        }
    }
}

/* ----------------------------------------------------------------- color */

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct HslOffset {
    /// -100..100, CapCut scale: +-100 shifts the colour +-30 deg towards its neighbour
    pub h: f64,
    /// -100..100
    pub s: f64,
    /// -100..100
    pub l: f64,
}

/// `Record<HslChannel, HslOffset>`; missing channels default to zero offsets.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct HslAdjustments {
    pub red: HslOffset,
    pub orange: HslOffset,
    pub yellow: HslOffset,
    pub green: HslOffset,
    pub cyan: HslOffset,
    pub blue: HslOffset,
    pub purple: HslOffset,
    pub magenta: HslOffset,
}

fn identity_curve() -> Vec<CurvePoint> {
    vec![[0.0, 0.0], [1.0, 1.0]]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ColorCurves {
    pub master: Vec<CurvePoint>,
    pub r: Vec<CurvePoint>,
    pub g: Vec<CurvePoint>,
    pub b: Vec<CurvePoint>,
}

impl Default for ColorCurves {
    fn default() -> Self {
        Self {
            master: identity_curve(),
            r: identity_curve(),
            g: identity_curve(),
            b: identity_curve(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
/// Adjust sliders use CapCut's scale: -50..50 (sharpness 0..50). HSL offsets use -100..100.
pub struct ColorGrade {
    pub exposure: f64,
    /// CapCut-style brilliance: lifts shadows and recovers highlights while keeping midtone contrast
    pub brilliance: f64,
    pub contrast: f64,
    pub brightness: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub saturation: f64,
    pub vibrance: f64,
    pub sharpness: f64,
    /// blue(-) .. yellow(+)
    pub temperature: f64,
    /// green(-) .. magenta(+)
    pub tint: f64,
    pub lift: Vec3,
    pub gamma: Vec3,
    pub gain: Vec3,
    pub offset: Vec3,
    pub hsl: HslAdjustments,
    pub curves: ColorCurves,
    pub lut_asset_id: Option<String>,
    /// 0..1
    pub lut_intensity: f64,
    /// 0..1
    pub vignette: f64,
    /// 0..1
    pub grain: f64,
}

impl Default for ColorGrade {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            brilliance: 0.0,
            contrast: 0.0,
            brightness: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            saturation: 0.0,
            vibrance: 0.0,
            sharpness: 0.0,
            temperature: 0.0,
            tint: 0.0,
            lift: [0.0; 3],
            gamma: [0.0; 3],
            gain: [0.0; 3],
            offset: [0.0; 3],
            hsl: HslAdjustments::default(),
            curves: ColorCurves::default(),
            lut_asset_id: None,
            lut_intensity: 1.0,
            vignette: 0.0,
            grain: 0.0,
        }
    }
}

impl ColorGrade {
    /// True when every control is at its neutral value.
    pub fn is_neutral(&self) -> bool {
        *self == ColorGrade::default()
    }
}

/* ----------------------------------------------------------------- audio */

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClipAudio {
    pub gain_db: f64,
    pub normalize: bool,
    pub muted: bool,
    /// CapCut-style voice separation: which part of the asset's sound the clip plays.
    pub voice: VoiceMode,
    /// Keep the pitch when the clip plays faster / slower (time-stretch instead of varispeed).
    /// Absent = `true` (CapCut "Pitch" off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_pitch: Option<bool>,
    /// Equal-power fade-in over the clip's first `fadeInMs` (timeline ms, clamped to half the clip).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fade_in_ms: Option<f64>,
    /// Equal-power fade-out over the clip's last `fadeOutMs` (timeline ms, clamped to half the clip).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fade_out_ms: Option<f64>,
    /// dB offset added to `gainDb`, keyframed in clip-local timeline ms (static value 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<Keyframed<f64>>,
}

impl Default for ClipAudio {
    fn default() -> Self {
        Self {
            gain_db: 0.0,
            normalize: true,
            muted: false,
            voice: VoiceMode::Original,
            keep_pitch: None,
            fade_in_ms: None,
            fade_out_ms: None,
            volume: None,
        }
    }
}

impl ClipAudio {
    /// `keepPitch`, default `true`.
    pub fn keep_pitch(&self) -> bool {
        self.keep_pitch.unwrap_or(true)
    }
}

/// `ClipAudio.voice`: `original` (unchanged), `voice` ("Isolate voice": the vocals stem) or
/// `background` ("Remove vocals": the background stem). Unknown values load as `original`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceMode {
    Voice,
    Background,
    // last: `#[serde(other)]` must be on the final variant
    #[default]
    #[serde(other)]
    Original,
}

/* ------------------------------------------------------------------ mask */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskShape {
    #[default]
    Rectangle,
    Circle,
    Split,
    Filmstrip,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClipMask {
    pub shape: MaskShape,
    /// 0..1
    pub feather: f64,
    /// normalised 0..1 of the frame
    pub rect: Keyframed<Rect>,
    pub inverted: bool,
}

impl Default for ClipMask {
    fn default() -> Self {
        Self {
            shape: MaskShape::Rectangle,
            feather: 0.1,
            rect: Keyframed::constant([0.0, 0.0, 1.0, 1.0]),
            inverted: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    SoftLight,
    Darken,
    Lighten,
    ColorDodge,
}

/* --------------------------------------------------------------- reframe */

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ReframeKeyframe {
    pub frame: u64,
    pub time_ms: f64,
    pub crop: BBox,
    pub zoom: f64,
    /// normalised translation -1..1
    pub tx: f64,
    pub ty: f64,
}

impl Default for ReframeKeyframe {
    fn default() -> Self {
        Self { frame: 0, time_ms: 0.0, crop: [0.0; 4], zoom: 1.0, tx: 0.0, ty: 0.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ReframeTrack {
    pub source_width: u32,
    pub source_height: u32,
    pub keyframes: Vec<ReframeKeyframe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FreezeFrame {
    pub at_ms: f64,
    pub hold_ms: f64,
}

/* ------------------------------------------------------------------ clip */

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Clip {
    pub id: String,
    pub asset_id: String,
    pub track_id: String,
    /// position on the timeline
    pub start_ms: f64,
    /// source range (before speed)
    pub in_ms: f64,
    pub out_ms: f64,
    pub speed: SpeedCurve,
    pub transform: ClipTransform,
    pub color: ColorGrade,
    pub audio: ClipAudio,
    pub mask: Option<ClipMask>,
    pub blend_mode: BlendMode,
    pub reframe: Option<ReframeTrack>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub freeze_frame: Option<FreezeFrame>,
    pub reversed: bool,
    /// Linked clips (a video clip, its mirrored audio clip and its stem clips) share the same
    /// id; a group may have any number of members.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_id: Option<String>,
    /// Transition from the previous clip on the same video track (the one ending where this clip
    /// starts, gap < 1 frame), centred on the cut. `null` / absent = a hard cut.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transition_in: Option<TransitionIn>,
    /// Set on clips of `fx` tracks (`assetId: ''`): the effect applied to the composited frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effect: Option<ClipEffect>,
    /// Video fade from black over the clip's first `fadeInMs` (clip-local timeline ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fade_in_ms: Option<f64>,
    /// Video fade to black over the clip's last `fadeOutMs` (clip-local timeline ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fade_out_ms: Option<f64>,
}

string_enum! {
    /// `Clip.transitionIn.type` (the maths is in `render::transitions`). Unknown types render as
    /// `dissolve`.
    TransitionType {
        Dissolve = "dissolve",
        DipToBlack = "dipToBlack",
        DipToWhite = "dipToWhite",
        WipeLeft = "wipeLeft",
        WipeRight = "wipeRight",
        WipeUp = "wipeUp",
        WipeDown = "wipeDown",
        SlideLeft = "slideLeft",
        SlideRight = "slideRight",
        PushLeft = "pushLeft",
        PushRight = "pushRight",
        ZoomIn = "zoomIn",
        ZoomOut = "zoomOut",
        BlurDissolve = "blurDissolve",
        Flash = "flash",
        CircleOpen = "circleOpen",
    }
}

#[allow(clippy::derivable_impls)] // the variants come from a macro
impl Default for TransitionType {
    fn default() -> Self {
        TransitionType::Dissolve
    }
}

/// Transition length limits (ms): `durationMs` is clamped to this range and to the shorter clip.
pub const TRANSITION_MIN_MS: f64 = 100.0;
pub const TRANSITION_MAX_MS: f64 = 3000.0;
pub const TRANSITION_DEFAULT_MS: f64 = 500.0;

fn default_transition_ms() -> f64 {
    TRANSITION_DEFAULT_MS
}

/// `Clip.transitionIn`: `{ type, durationMs }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransitionIn {
    #[serde(rename = "type", default)]
    pub kind: TransitionType,
    #[serde(default = "default_transition_ms")]
    pub duration_ms: f64,
}

string_enum! {
    /// `Clip.effect.type` (the maths is in `render::fx`). Unknown types are skipped (no-op).
    EffectType {
        CameraSnap = "cameraSnap",
        FadeFromBlack = "fadeFromBlack",
        FadeToBlack = "fadeToBlack",
        FadeFromWhite = "fadeFromWhite",
        FadeToWhite = "fadeToWhite",
        BlackAndWhite = "blackAndWhite",
        Sepia = "sepia",
        Letterbox = "letterbox",
        Shake = "shake",
        ZoomPunch = "zoomPunch",
        BlurIn = "blurIn",
        BlurOut = "blurOut",
        RgbSplit = "rgbSplit",
        Vhs = "vhs",
        VignettePulse = "vignettePulse",
        FlashWhite = "flashWhite",
    }
}

impl Default for EffectType {
    fn default() -> Self {
        EffectType::Other(String::new())
    }
}

fn default_intensity() -> f64 {
    1.0
}

/// `Clip.effect`: `{ type, intensity (0..1, default 1), params? }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipEffect {
    #[serde(rename = "type", default)]
    pub kind: EffectType,
    #[serde(default = "default_intensity")]
    pub intensity: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<std::collections::BTreeMap<String, f64>>,
}

impl ClipEffect {
    pub fn new(kind: EffectType) -> Self {
        Self { kind, intensity: 1.0, params: None }
    }

    /// A numeric param, or `default` when absent / not finite.
    pub fn param(&self, name: &str, default: f64) -> f64 {
        self.params.as_ref().and_then(|p| p.get(name)).copied().filter(|v| v.is_finite()).unwrap_or(default)
    }
}

impl Clip {
    /// Length of the source range in ms (before speed remapping).
    pub fn source_duration_ms(&self) -> f64 {
        (self.out_ms - self.in_ms).max(0.0)
    }

    /// Length on the timeline after speed remapping (+ freeze-frame hold).
    pub fn output_duration_ms(&self) -> f64 {
        let base = self.speed.lut(self.source_duration_ms()).output_duration_ms();
        base + self.freeze_frame.map(|f| f.hold_ms.max(0.0)).unwrap_or(0.0)
    }
}

/* --------------------------------------------------------------- project */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BeatKind {
    #[default]
    Beat1,
    Beat2,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct BeatMarker {
    pub time_ms: f64,
    /// 0..1
    pub strength: f64,
    pub kind: BeatKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Project {
    pub version: u32,
    pub id: String,
    pub name: String,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub assets: Vec<Asset>,
    pub tracks: Vec<Track>,
    pub beat_markers: Vec<BeatMarker>,
    /// Project-wide adjustment layered on top of every clip's own grade (toggleable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub universal_adjust: Option<UniversalAdjust>,
    /// How the exporter makes frames the sources do not have (output fps above a clip's
    /// effective fps). Absent = `opticalFlow`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_interpolation: Option<FrameInterpolation>,
}

string_enum! {
    /// `Project.frameInterpolation`: `none` repeats the nearest source frame, `frameBlend`
    /// crossfades the two neighbouring source frames, `opticalFlow` synthesises in-between frames
    /// with the pipeline's `interpolate --target-fps` (falling back to `frameBlend`). Unknown
    /// values behave as `opticalFlow`, the default.
    FrameInterpolation { None = "none", FrameBlend = "frameBlend", OpticalFlow = "opticalFlow" }
}

/// Project frame rates the UI offers / the exporter accepts (±0.01): 24, 25, 30, 40, 48, 50, 60,
/// plus 23.976, 29.97 and 59.94 (source rates; the UI snaps sources to these).
pub const ALLOWED_FPS: &[f64] = &[23.976, 24.0, 25.0, 29.97, 30.0, 40.0, 48.0, 50.0, 59.94, 60.0];

/// The adjustable scalar sliders of a [`ColorGrade`]; `None` = leave the clip's value alone.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AdjustValues {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brilliance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brightness: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlights: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadows: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibrance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sharpness: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tint: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vignette: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grain: Option<f64>,
    /// per-channel HSL offsets added to the clip's
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hsl: Option<HslAdjustments>,
}

/// A toggleable adjustment applied to every clip of a project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct UniversalAdjust {
    pub enabled: bool,
    pub name: String,
    pub values: AdjustValues,
}

impl Default for UniversalAdjust {
    fn default() -> Self {
        Self { enabled: true, name: "Universal adjust".into(), values: AdjustValues::default() }
    }
}

/// The grade a clip is rendered with: its own grade plus the project's universal adjustment
/// (when enabled). Sliders add and are clamped to their ranges; everything else (wheels, HSL,
/// curves, LUT) comes from the clip. Mirrors `effectiveGrade` in `src/engine/grade.ts`.
pub fn effective_grade(clip: &ColorGrade, universal: Option<&UniversalAdjust>) -> ColorGrade {
    let Some(u) = universal.filter(|u| u.enabled) else {
        return clip.clone();
    };
    let v = &u.values;
    let mut g = clip.clone();
    // Only fields with a delta are touched (and clamped), like `effectiveGrade` in grade.ts.
    let add = |base: f64, delta: Option<f64>, lo: f64, hi: f64| match delta {
        Some(d) => (base + d).clamp(lo, hi),
        None => base,
    };
    g.exposure = add(g.exposure, v.exposure, -50.0, 50.0);
    g.brilliance = add(g.brilliance, v.brilliance, -50.0, 50.0);
    g.contrast = add(g.contrast, v.contrast, -50.0, 50.0);
    g.brightness = add(g.brightness, v.brightness, -50.0, 50.0);
    g.highlights = add(g.highlights, v.highlights, -50.0, 50.0);
    g.shadows = add(g.shadows, v.shadows, -50.0, 50.0);
    g.saturation = add(g.saturation, v.saturation, -50.0, 50.0);
    g.vibrance = add(g.vibrance, v.vibrance, -50.0, 50.0);
    g.sharpness = add(g.sharpness, v.sharpness, 0.0, 50.0);
    g.temperature = add(g.temperature, v.temperature, -50.0, 50.0);
    g.tint = add(g.tint, v.tint, -50.0, 50.0);
    g.vignette = add(g.vignette, v.vignette, 0.0, 1.0);
    g.grain = add(g.grain, v.grain, 0.0, 1.0);
    if let Some(h) = &v.hsl {
        let mix = |a: &mut HslOffset, b: &HslOffset| {
            if *b == HslOffset::default() {
                return; // a missing / zero channel in the preset leaves the clip's channel alone
            }
            a.h = (a.h + b.h).clamp(-100.0, 100.0);
            a.s = (a.s + b.s).clamp(-100.0, 100.0);
            a.l = (a.l + b.l).clamp(-100.0, 100.0);
        };
        mix(&mut g.hsl.red, &h.red);
        mix(&mut g.hsl.orange, &h.orange);
        mix(&mut g.hsl.yellow, &h.yellow);
        mix(&mut g.hsl.green, &h.green);
        mix(&mut g.hsl.cyan, &h.cyan);
        mix(&mut g.hsl.blue, &h.blue);
        mix(&mut g.hsl.purple, &h.purple);
        mix(&mut g.hsl.magenta, &h.magenta);
    }
    g
}

impl Default for Project {
    fn default() -> Self {
        Self {
            version: 1,
            id: String::new(),
            name: String::new(),
            fps: 24.0,
            width: 1920,
            height: 1080,
            assets: Vec::new(),
            tracks: Vec::new(),
            beat_markers: Vec::new(),
            universal_adjust: None,
            frame_interpolation: None,
        }
    }
}

/// Limits a project must respect to be rendered (see [`Project::validate`]).
pub const MAX_DIMENSION: u32 = 8192;
pub const MIN_DIMENSION: u32 = 16;
pub const MAX_FPS: f64 = 240.0;
/// 4 hours of timeline.
pub const MAX_DURATION_MS: f64 = 4.0 * 3600.0 * 1000.0;
pub const MAX_TRACKS: usize = 256;
pub const MAX_CLIPS: usize = 50_000;

impl Project {
    pub fn asset(&self, id: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.id == id)
    }

    /// `frameInterpolation` with its default (`opticalFlow`; unknown values too).
    pub fn frame_interpolation(&self) -> FrameInterpolation {
        match &self.frame_interpolation {
            Some(f @ (FrameInterpolation::None | FrameInterpolation::FrameBlend)) => f.clone(),
            _ => FrameInterpolation::OpticalFlow,
        }
    }

    /// Linked clip groups: `linkId` → ids of every clip carrying it (any number of members, in
    /// track order).
    pub fn link_groups(&self) -> std::collections::BTreeMap<String, Vec<String>> {
        let mut m: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        for c in self.tracks.iter().flat_map(|t| t.clips.iter()) {
            if let Some(l) = &c.link_id {
                m.entry(l.clone()).or_default().push(c.id.clone());
            }
        }
        m
    }

    /// Is `fps` one of [`ALLOWED_FPS`] (±0.01)?
    pub fn is_allowed_fps(fps: f64) -> bool {
        ALLOWED_FPS.iter().any(|a| (a - fps).abs() <= 0.01)
    }

    /// Reject documents whose numbers would size buffers absurdly (canvas, per-frame plan,
    /// audio mix) or that cannot be rendered: `width`/`height` in 16..=8192, `fps` one of
    /// [`ALLOWED_FPS`], finite clip times, fades, transition / effect parameters and volume
    /// keyframes, a timeline of at most 4 h, at most 256 tracks / 50 000 clips. (Out-of-range but
    /// finite transition / fade lengths are clamped when rendering, not rejected.)
    pub fn validate(&self) -> Result<(), String> {
        if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&self.width) || !(MIN_DIMENSION..=MAX_DIMENSION).contains(&self.height) {
            return Err(format!(
                "project size {}x{} is outside {MIN_DIMENSION}..{MAX_DIMENSION} pixels",
                self.width, self.height
            ));
        }
        if !(self.fps.is_finite() && self.fps > 1.0 && self.fps <= MAX_FPS) || !Self::is_allowed_fps(self.fps) {
            return Err(format!(
                "project frame rate {} is not supported (use 24, 25, 30, 40, 48, 50 or 60; 23.976 / 29.97 / 59.94 for sources)",
                self.fps
            ));
        }
        if self.tracks.len() > MAX_TRACKS {
            return Err(format!("{} tracks (at most {MAX_TRACKS})", self.tracks.len()));
        }
        let clips = self.tracks.iter().map(|t| t.clips.len()).sum::<usize>();
        if clips > MAX_CLIPS {
            return Err(format!("{clips} clips (at most {MAX_CLIPS})"));
        }
        let mut end = 0.0f64;
        for c in self.tracks.iter().flat_map(|t| t.clips.iter()) {
            let finite = [c.start_ms, c.in_ms, c.out_ms].iter().all(|v| v.is_finite() && v.abs() <= MAX_DURATION_MS * 4.0)
                && c.freeze_frame.map(|f| f.at_ms.is_finite() && f.hold_ms.is_finite() && f.hold_ms <= MAX_DURATION_MS).unwrap_or(true)
                && c.speed.points.iter().all(|p| p.t.is_finite() && p.speed.is_finite() && p.speed > 0.0);
            if !finite {
                return Err(format!("clip {} has invalid times or speed points", c.id));
            }
            let sane_ms = |v: Option<f64>| v.map(|x| x.is_finite() && (0.0..=MAX_DURATION_MS).contains(&x)).unwrap_or(true);
            let a = &c.audio;
            let v2_ok = sane_ms(c.fade_in_ms)
                && sane_ms(c.fade_out_ms)
                && sane_ms(a.fade_in_ms)
                && sane_ms(a.fade_out_ms)
                && a.volume.as_ref().map(|k| {
                    k.static_value.is_finite() && k.keyframes.iter().all(|kf| kf.time_ms.is_finite() && kf.value.is_finite() && kf.value.abs() <= 200.0)
                }).unwrap_or(true)
                && c.transition_in.as_ref().map(|t| t.duration_ms.is_finite() && t.duration_ms >= 0.0).unwrap_or(true)
                && c.effect.as_ref().map(|e| {
                    e.intensity.is_finite() && e.params.as_ref().map(|p| p.values().all(|v| v.is_finite())).unwrap_or(true)
                }).unwrap_or(true);
            if !v2_ok {
                return Err(format!("clip {} has an invalid fade, volume keyframe, transition or effect value", c.id));
            }
            if c.source_duration_ms() > 0.0 {
                end = end.max(c.start_ms + c.output_duration_ms());
            }
        }
        if !(end.is_finite() && end <= MAX_DURATION_MS) {
            return Err(format!(
                "the timeline is {:.1} h long; at most {:.0} h can be rendered",
                end / 3_600_000.0,
                MAX_DURATION_MS / 3_600_000.0
            ));
        }
        Ok(())
    }

    /// The first video track (export target for phase 1).
    pub fn first_video_track(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.kind == TrackKind::Video)
    }
}

/* ------------------------------------------------- pipeline analysis result */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShotMethod {
    #[default]
    Transnetv2,
    Pyscenedetect,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Shot {
    pub index: u32,
    pub start_frame: u64,
    pub end_frame: u64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub confidence: f64,
    pub method: ShotMethod,
    /// character ids visible in the shot
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DetectedInstance {
    pub track_id: i64,
    pub label: String,
    pub bbox: BBox,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DuplicateFinding {
    pub shot_index: u32,
    pub frame: u64,
    pub time_ms: f64,
    pub primary: DetectedInstance,
    pub duplicate: DetectedInstance,
    pub similarity: f64,
    /// character id the duplicate was matched to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub character: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub character_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ShotReframe {
    pub shot_index: u32,
    pub track: ReframeTrack,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AudioAnalysis {
    pub integrated_lufs: f64,
    pub true_peak_db: f64,
    pub recommended_gain_db: f64,
    /// ms
    pub beats: Vec<f64>,
    pub tempo_bpm: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransitionSuggestion {
    #[default]
    Cut,
    Dissolve,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TransitionAnalysis {
    pub from_shot: u32,
    pub to_shot: u32,
    pub flow_magnitude: f64,
    /// 0..1
    pub smoothness: f64,
    pub suggestion: TransitionSuggestion,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClipAnalysis {
    pub path: String,
    pub asset: Asset,
    pub shots: Vec<Shot>,
    pub duplicates: Vec<DuplicateFinding>,
    pub reframe: Vec<ShotReframe>,
    pub audio: Option<AudioAnalysis>,
    pub transitions: Vec<TransitionAnalysis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AnalysisResult {
    pub version: u32,
    pub generated_at: String,
    pub clips: Vec<ClipAnalysis>,
    pub timeline: Project,
}

impl Default for AnalysisResult {
    fn default() -> Self {
        Self { version: 1, generated_at: String::new(), clips: Vec::new(), timeline: Project::default() }
    }
}

/* ------------------------------------------------- pipeline progress events */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PipelineStage {
    #[default]
    Ingest,
    Shots,
    Perception,
    Reframe,
    Audio,
    Transitions,
    Assemble,
    Export,
    /// Any stage name this build does not know (forward compatibility).
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    #[default]
    Info,
    Warn,
    Error,
}

/// One line of pipeline stdout, tagged by `"event"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum PipelineEvent {
    #[serde(rename_all = "camelCase")]
    Progress {
        #[serde(default)]
        stage: PipelineStage,
        #[serde(default)]
        clip: Option<String>,
        #[serde(default)]
        pct: f64,
        #[serde(default)]
        message: String,
    },
    #[serde(rename_all = "camelCase")]
    Log {
        #[serde(default)]
        level: LogLevel,
        #[serde(default)]
        message: String,
    },
    #[serde(rename_all = "camelCase")]
    Result { path: String },
}

/* ------------------------------------------------------- pipeline options */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Detector {
    YoloWorld,
    GroundedSam2,
    #[default]
    Hybrid,
}

impl Detector {
    pub fn as_cli(self) -> &'static str {
        match self {
            Detector::YoloWorld => "yolo_world",
            Detector::GroundedSam2 => "grounded_sam2",
            Detector::Hybrid => "hybrid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShotDetector {
    Transnetv2,
    Pyscenedetect,
    #[default]
    Auto,
}

impl ShotDetector {
    pub fn as_cli(self) -> &'static str {
        match self {
            ShotDetector::Transnetv2 => "transnetv2",
            ShotDetector::Pyscenedetect => "pyscenedetect",
            ShotDetector::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Smoothing {
    #[default]
    Ema,
    Savgol,
}

impl Smoothing {
    pub fn as_cli(self) -> &'static str {
        match self {
            Smoothing::Ema => "ema",
            Smoothing::Savgol => "savgol",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PipelineOptions {
    /// open-vocabulary prompts, e.g. ["raccoon in hoodie", "person"]
    pub prompts: Vec<String>,
    pub detector: Detector,
    pub shot_detector: ShotDetector,
    /// e.g. 165000 (2:45)
    pub target_duration_ms: f64,
    pub normalize_audio: bool,
    /// e.g. -14
    pub target_lufs: f64,
    pub detect_beats: bool,
    /// 0..1, default 0.85
    pub similarity_threshold: f64,
    pub smoothing: Smoothing,
    /// EMA inertia 0..1
    pub smoothing_alpha: f64,
    /// keep the given clip order instead of the pipeline's ordering heuristics
    pub keep_order: bool,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            prompts: Vec::new(),
            detector: Detector::Hybrid,
            shot_detector: ShotDetector::Auto,
            target_duration_ms: 165_000.0,
            normalize_audio: true,
            target_lufs: -14.0,
            detect_beats: true,
            similarity_threshold: 0.85,
            smoothing: Smoothing::Ema,
            smoothing_alpha: 0.15,
            keep_order: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    const SAMPLE_PROJECT: &str = r#"{
      "version": 1, "id": "proj_1", "name": "Ep_01", "fps": 24, "width": 1920, "height": 1080,
      "assets": [{ "id": "ast_1", "path": "C:/clips/Clip_01.mp4", "name": "Clip_01.mp4", "kind": "video",
                   "durationMs": 50000, "width": 1920, "height": 1080, "fps": 24, "hasAudio": true,
                   "codec": "h264", "sceneTags": ["kitchen","raccoon"], "order": 0, "orderReason": "leading number 01",
                   "stems": { "vocals": "C:/cache/stems/ab/vocals.wav", "background": "C:/cache/stems/ab/background.wav" } }],
      "tracks": [{ "id": "trk_1", "kind": "video", "name": "Video 1", "locked": false, "muted": false,
        "clips": [{
          "id": "clp_1", "assetId": "ast_1", "trackId": "trk_1", "startMs": 0, "inMs": 0, "outMs": 5000,
          "speed": { "preset": "hero_time", "points": [{"t":0,"speed":1},{"t":0.5,"speed":0.3},{"t":1,"speed":1}], "opticalFlow": true },
          "transform": { "position": { "static": [0,0], "keyframes": [] },
                         "scale": { "static": 1, "keyframes": [ { "timeMs": 0, "value": 1, "easing": "easeInOut" },
                                                                { "timeMs": 1000, "value": 1.5, "easing": "bezier", "bezier": [0.2,0,0.8,1] } ] },
                         "rotation": { "static": 0, "keyframes": [] },
                         "opacity": { "static": 1, "keyframes": [] },
                         "blur": { "static": 0, "keyframes": [] } },
          "color": { "exposure": 0, "brilliance": 0, "contrast": 10, "brightness": 0, "highlights": 0, "shadows": 0, "saturation": 5,
                     "vibrance": 0, "sharpness": 0, "temperature": 0, "tint": 0,
                     "lift": [0,0,0], "gamma": [0,0,0], "gain": [0,0,0], "offset": [0,0,0],
                     "hsl": { "red": {"h":0,"s":0,"l":0}, "orange": {"h":0,"s":0,"l":0}, "yellow": {"h":0,"s":0,"l":0},
                              "green": {"h":0,"s":0,"l":0}, "cyan": {"h":0,"s":0,"l":0}, "blue": {"h":0,"s":0,"l":0},
                              "purple": {"h":0,"s":0,"l":0}, "magenta": {"h":0,"s":0,"l":0} },
                     "curves": { "master": [[0,0],[1,1]], "r": [[0,0],[1,1]], "g": [[0,0],[1,1]], "b": [[0,0],[1,1]] },
                     "lutAssetId": null, "lutIntensity": 1.0, "vignette": 0, "grain": 0 },
          "audio": { "gainDb": 3, "normalize": true, "muted": false, "voice": "voice" },
          "mask": { "shape": "circle", "feather": 0.1, "rect": { "static": [0.1,0.1,0.8,0.8], "keyframes": [] }, "inverted": false },
          "blendMode": "softLight",
          "reframe": { "sourceWidth": 1920, "sourceHeight": 1080,
                       "keyframes": [ { "frame": 0, "timeMs": 0, "crop": [100,0,1500,1080], "zoom": 1.45, "tx": -0.12, "ty": 0.0 } ],
                       "reason": "duplicate raccoon excluded on right boundary" },
          "label": "Shot 2", "freezeFrame": { "atMs": 1200, "holdMs": 800 }, "reversed": false,
          "linkId": "lnk_1"
        }]
      }, { "id": "trk_2", "kind": "audio", "name": "Audio 1", "locked": false, "muted": false,
        "clips": [{
          "id": "clp_2", "assetId": "ast_1", "trackId": "trk_2", "startMs": 0, "inMs": 0, "outMs": 5000,
          "speed": { "preset": "normal", "points": [], "opticalFlow": false },
          "transform": { "position": { "static": [0,0], "keyframes": [] }, "scale": { "static": 1, "keyframes": [] },
                         "rotation": { "static": 0, "keyframes": [] }, "opacity": { "static": 1, "keyframes": [] },
                         "blur": { "static": 0, "keyframes": [] } },
          "color": { "exposure": 0, "brilliance": 0, "contrast": 0, "brightness": 0, "highlights": 0, "shadows": 0, "saturation": 0,
                     "vibrance": 0, "sharpness": 0, "temperature": 0, "tint": 0,
                     "lift": [0,0,0], "gamma": [0,0,0], "gain": [0,0,0], "offset": [0,0,0],
                     "hsl": { "red": {"h":0,"s":0,"l":0}, "orange": {"h":0,"s":0,"l":0}, "yellow": {"h":0,"s":0,"l":0},
                              "green": {"h":0,"s":0,"l":0}, "cyan": {"h":0,"s":0,"l":0}, "blue": {"h":0,"s":0,"l":0},
                              "purple": {"h":0,"s":0,"l":0}, "magenta": {"h":0,"s":0,"l":0} },
                     "curves": { "master": [[0,0],[1,1]], "r": [[0,0],[1,1]], "g": [[0,0],[1,1]], "b": [[0,0],[1,1]] },
                     "lutAssetId": null, "lutIntensity": 1.0, "vignette": 0, "grain": 0 },
          "audio": { "gainDb": 0, "normalize": true, "muted": false, "voice": "voice" },
          "mask": null, "blendMode": "normal", "reframe": null, "freezeFrame": null, "reversed": false,
          "linkId": "lnk_1"
        }]
      }],
      "beatMarkers": [ { "timeMs": 1234.5, "strength": 0.8, "kind": "beat1" } ]
    }"#;

    /// serde_json's `Value` distinguishes `1` (u64) from `1.0` (f64); JSON does not.
    fn norm(v: Value) -> Value {
        match v {
            Value::Number(n) => json!(n.as_f64().unwrap_or(0.0)),
            Value::Array(a) => Value::Array(a.into_iter().map(norm).collect()),
            Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, norm(v))).collect()),
            other => other,
        }
    }

    /// Does `sup` contain every key / element of `sub` with the same value?
    fn json_subset(sup: &Value, sub: &Value) -> bool {
        match (sup, sub) {
            (Value::Object(a), Value::Object(b)) => b.iter().all(|(k, v)| a.get(k).map(|x| json_subset(x, v)).unwrap_or(false)),
            (Value::Array(a), Value::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_subset(x, y)),
            (a, b) => a == b,
        }
    }

    #[test]
    fn project_round_trip_preserves_every_field() {
        let project: Project = serde_json::from_str(SAMPLE_PROJECT).expect("parse");
        let original: Value = serde_json::from_str(SAMPLE_PROJECT).unwrap();
        let re: Value = serde_json::to_value(&project).unwrap();
        assert_eq!(norm(re), norm(original), "serialised JSON must match the source document");

        let clip = &project.tracks[0].clips[0];
        assert_eq!(clip.blend_mode, BlendMode::SoftLight);
        assert_eq!(clip.speed.preset, SpeedPreset::HeroTime);
        assert_eq!(clip.transform.scale.keyframes[1].easing, Easing::Bezier);
        assert_eq!(clip.mask.as_ref().unwrap().shape, MaskShape::Circle);
        assert_eq!(project.beat_markers[0].kind, BeatKind::Beat1);
        assert_eq!(project.assets[0].kind, AssetKind::Video);
        assert!(clip.output_duration_ms() > 5000.0 + 800.0);
        assert_eq!(clip.audio.voice, VoiceMode::Voice);
        assert_eq!(project.assets[0].stems.as_ref().unwrap().background, "C:/cache/stems/ab/background.wav");
        // linked video / audio clips share a linkId; clips without one don't serialise it
        assert_eq!(clip.link_id.as_deref(), Some("lnk_1"));
        assert_eq!(project.tracks[1].clips[0].link_id, clip.link_id);
        assert!(serde_json::to_value(Clip::default()).unwrap().get("linkId").is_none());
    }

    #[test]
    fn project_validation_limits() {
        let clip = |start: f64, out: f64| Clip { id: "c".into(), start_ms: start, in_ms: 0.0, out_ms: out, ..Default::default() };
        let ok = Project { tracks: vec![Track { clips: vec![clip(0.0, 1000.0)], ..Default::default() }], ..Default::default() };
        assert!(ok.validate().is_ok());
        let with = |f: &dyn Fn(&mut Project)| {
            let mut p = ok.clone();
            f(&mut p);
            p.validate()
        };
        assert!(with(&|p| p.width = 8192).is_ok());
        assert!(with(&|p| p.width = 8194).unwrap_err().contains("8192"));
        assert!(with(&|p| p.height = 0).is_err());
        assert!(with(&|p| p.height = 100_000).is_err());
        for fps in [24.0, 25.0, 30.0, 40.0, 48.0, 50.0, 60.0, 23.976, 24000.0 / 1001.0, 29.97, 30000.0 / 1001.0, 59.94, 60000.0 / 1001.0] {
            assert!(with(&|p| p.fps = fps).is_ok(), "{fps} fps is allowed");
        }
        for fps in [240.0, 12.0, 45.0, 59.0, 100.0] {
            assert!(with(&|p| p.fps = fps).unwrap_err().contains("not supported"), "{fps} fps is refused");
        }
        assert!(with(&|p| p.fps = 1.0).is_err(), "fps must be above 1");
        assert!(with(&|p| p.fps = 0.0).is_err());
        assert!(with(&|p| p.fps = 1000.0).is_err());
        assert!(with(&|p| p.fps = f64::NAN).is_err());
        assert!(with(&|p| p.tracks[0].clips[0].start_ms = f64::INFINITY).is_err());
        assert!(with(&|p| p.tracks[0].clips[0].out_ms = f64::NAN).is_err());
        assert!(with(&|p| p.tracks[0].clips[0].start_ms = MAX_DURATION_MS).unwrap_err().contains("at most 4 h"));
        assert!(with(&|p| p.tracks[0].clips[0].freeze_frame = Some(FreezeFrame { at_ms: 0.0, hold_ms: 1e12 })).is_err());
        assert!(with(&|p| p.tracks[0].clips[0].speed = SpeedCurve::custom(vec![SpeedPoint::new(0.0, f64::NAN)])).is_err());
        assert!(with(&|p| p.tracks = vec![Track::default(); MAX_TRACKS + 1]).is_err());
        // v2 fields: finite, non-negative durations
        assert!(with(&|p| p.tracks[0].clips[0].fade_in_ms = Some(300.0)).is_ok());
        assert!(with(&|p| p.tracks[0].clips[0].fade_in_ms = Some(-1.0)).is_err());
        assert!(with(&|p| p.tracks[0].clips[0].audio.fade_out_ms = Some(f64::NAN)).is_err());
        assert!(with(&|p| p.tracks[0].clips[0].audio.volume = Some(Keyframed::with_keyframes(0.0, vec![Keyframe::new(0.0, f64::INFINITY)]))).is_err());
        assert!(with(&|p| p.tracks[0].clips[0].transition_in = Some(TransitionIn { kind: TransitionType::Flash, duration_ms: 5000.0 })).is_ok(), "clamped when rendering");
        assert!(with(&|p| p.tracks[0].clips[0].transition_in = Some(TransitionIn { kind: TransitionType::Flash, duration_ms: f64::NAN })).is_err());
        assert!(with(&|p| {
            let mut e = ClipEffect::new(EffectType::Shake);
            e.params = Some([("amplitude".to_string(), f64::NAN)].into_iter().collect());
            p.tracks[0].clips[0].effect = Some(e);
        })
        .is_err());
    }

    #[test]
    fn v2_fields_round_trip_and_default() {
        let doc = json!({
            "version": 1, "id": "p", "name": "v2", "fps": 60, "width": 1280, "height": 720,
            "frameInterpolation": "frameBlend",
            "assets": [
                { "id": "v", "path": "C:/a.mp4", "name": "a.mp4", "kind": "video", "durationMs": 9000, "width": 1280, "height": 720, "fps": 24, "hasAudio": true },
                { "id": "s1", "path": "C:/stems/vocals.wav", "name": "a · Voice", "kind": "audio", "durationMs": 9000, "width": 0, "height": 0, "fps": 0, "hasAudio": true,
                  "stemOf": { "assetId": "v", "stem": "vocals" } }
            ],
            "tracks": [
                { "id": "t1", "kind": "video", "name": "Video 1", "locked": false, "muted": false, "clips": [
                    { "id": "c1", "assetId": "v", "trackId": "t1", "startMs": 0, "inMs": 0, "outMs": 3000, "linkId": "lnk_1",
                      "fadeInMs": 250, "fadeOutMs": 400,
                      "audio": { "gainDb": -3, "normalize": false, "muted": true, "voice": "original", "keepPitch": false,
                                 "fadeInMs": 120, "fadeOutMs": 800,
                                 "volume": { "static": 0, "keyframes": [ { "timeMs": 0, "value": -6, "easing": "easeInOut" }, { "timeMs": 1000, "value": 3, "easing": "linear" } ] } } },
                    { "id": "c2", "assetId": "v", "trackId": "t1", "startMs": 3000, "inMs": 3000, "outMs": 6000,
                      "transitionIn": { "type": "dipToBlack", "durationMs": 800 } }
                ] },
                { "id": "fx", "kind": "fx", "name": "FX", "locked": false, "muted": false, "clips": [
                    { "id": "e1", "assetId": "", "trackId": "fx", "startMs": 1000, "inMs": 0, "outMs": 1500,
                      "effect": { "type": "cameraSnap", "intensity": 0.8, "params": { "border": 0.05, "scale": 0.9 } } }
                ] },
                { "id": "a1", "kind": "audio", "name": "Voice", "role": "voice", "locked": false, "muted": false, "clips": [
                    { "id": "c3", "assetId": "s1", "trackId": "a1", "startMs": 0, "inMs": 0, "outMs": 3000, "linkId": "lnk_1" },
                    { "id": "c4", "assetId": "s1", "trackId": "a1", "startMs": 0, "inMs": 0, "outMs": 3000, "linkId": "lnk_1" }
                ] }
            ],
            "beatMarkers": []
        });
        let p: Project = serde_json::from_value(doc.clone()).expect("parse v2");
        assert_eq!(p.frame_interpolation(), FrameInterpolation::FrameBlend);
        assert_eq!(p.assets[1].stem_of.as_ref().unwrap().stem, StemKind::Vocals);
        assert_eq!(p.tracks[2].role, Some(TrackRole::Voice));
        let c1 = &p.tracks[0].clips[0];
        assert!(!c1.audio.keep_pitch());
        assert_eq!((c1.fade_in_ms, c1.fade_out_ms), (Some(250.0), Some(400.0)));
        assert_eq!(c1.audio.volume.as_ref().unwrap().evaluate(0.0), -6.0);
        let t = p.tracks[0].clips[1].transition_in.as_ref().unwrap();
        assert_eq!((t.kind.clone(), t.duration_ms), (TransitionType::DipToBlack, 800.0));
        let e = p.tracks[1].clips[0].effect.as_ref().unwrap();
        assert_eq!(e.kind, EffectType::CameraSnap);
        assert_eq!((e.intensity, e.param("border", 0.03), e.param("missing", 7.0)), (0.8, 0.05, 7.0));
        // linked groups of more than two members
        assert_eq!(p.link_groups()["lnk_1"], vec!["c1", "c3", "c4"]);
        assert!(p.validate().is_ok());
        // round trip: every field of the document comes back with its value (defaults are added
        // for the fields the document omitted), and re-parsing gives the same project
        let out = serde_json::to_value(&p).unwrap();
        assert!(json_subset(&norm(out.clone()), &norm(doc)), "{out:#}");
        assert_eq!(serde_json::from_value::<Project>(out).unwrap(), p);

        // defaults: old documents load, and the new fields stay out of the JSON
        let old: Project = serde_json::from_str(r#"{"tracks":[{"id":"t","kind":"video","clips":[{"id":"c","assetId":"a","trackId":"t","inMs":0,"outMs":1000}]}]}"#).unwrap();
        assert_eq!(old.frame_interpolation(), FrameInterpolation::OpticalFlow);
        let c = &old.tracks[0].clips[0];
        assert!(c.audio.keep_pitch());
        assert!(c.transition_in.is_none() && c.effect.is_none() && c.fade_in_ms.is_none() && c.audio.volume.is_none());
        let v = serde_json::to_value(&old).unwrap();
        assert!(v.get("frameInterpolation").is_none(), "frameInterpolation omitted");
        let cv = &v["tracks"][0]["clips"][0];
        for key in ["transitionIn", "effect", "fadeInMs", "fadeOutMs"] {
            assert!(cv.get(key).is_none(), "{key} omitted");
        }
        assert!(v["tracks"][0].get("role").is_none());
        // a null transition is a hard cut
        let c: Clip = serde_json::from_str(r#"{"transitionIn":null}"#).unwrap();
        assert!(c.transition_in.is_none());
        // missing durationMs / intensity take their defaults
        let c: Clip = serde_json::from_str(r#"{"transitionIn":{"type":"wipeLeft"},"effect":{"type":"sepia"}}"#).unwrap();
        assert_eq!(c.transition_in.unwrap().duration_ms, TRANSITION_DEFAULT_MS);
        assert_eq!(c.effect.unwrap().intensity, 1.0);
    }

    #[test]
    fn unknown_enum_values_load_and_round_trip() {
        let c: Clip = serde_json::from_str(
            r#"{"transitionIn":{"type":"starWipe","durationMs":300},"effect":{"type":"hologram","intensity":0.5}}"#,
        )
        .expect("unknown types must not fail");
        let t = c.transition_in.as_ref().unwrap();
        assert_eq!(t.kind, TransitionType::Other("starWipe".into()));
        assert!(!t.kind.is_known());
        assert_eq!(c.effect.as_ref().unwrap().kind, EffectType::Other("hologram".into()));
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["transitionIn"]["type"], "starWipe", "kept verbatim");
        assert_eq!(v["effect"]["type"], "hologram");
        let p: Project = serde_json::from_str(r#"{"frameInterpolation":"magic"}"#).unwrap();
        assert_eq!(p.frame_interpolation(), FrameInterpolation::OpticalFlow, "unknown → default");
        let t: Track = serde_json::from_str(r#"{"role":"music"}"#).unwrap();
        assert_eq!(t.role, Some(TrackRole::Other("music".into())));
        let a: Asset = serde_json::from_str(r#"{"stemOf":{"assetId":"x","stem":"drums"}}"#).unwrap();
        assert_eq!(a.stem_of.unwrap().stem, StemKind::Other("drums".into()));
        // non-string values do not fail either
        let t: TransitionIn = serde_json::from_str(r#"{"type":7}"#).unwrap();
        assert_eq!(t.kind, TransitionType::Other("7".into()));
        // every known name parses back to itself
        for k in TransitionType::ALL {
            assert_eq!(&TransitionType::parse(k.as_str()), k);
        }
        assert_eq!(TransitionType::ALL.len(), 16);
        for k in EffectType::ALL {
            assert_eq!(&EffectType::parse(k.as_str()), k);
        }
        assert_eq!(EffectType::ALL.len(), 16);
    }

    #[test]
    fn effective_grade_only_clamps_fields_with_a_delta() {
        // an out-of-range clip value without a preset delta is left alone (grade.ts semantics)
        let clip = ColorGrade { contrast: 80.0, saturation: 45.0, ..Default::default() };
        let u = UniversalAdjust { enabled: true, name: "u".into(), values: AdjustValues { saturation: Some(10.0), ..Default::default() } };
        let g = effective_grade(&clip, Some(&u));
        assert_eq!(g.contrast, 80.0);
        assert_eq!(g.saturation, 50.0);
        let mut c2 = clip.clone();
        c2.hsl.red.h = 150.0;
        let u2 = UniversalAdjust {
            values: AdjustValues { hsl: Some(HslAdjustments { green: HslOffset { h: 10.0, s: 0.0, l: 0.0 }, ..Default::default() }), ..Default::default() },
            ..u
        };
        let g = effective_grade(&c2, Some(&u2));
        assert_eq!(g.hsl.red.h, 150.0, "channel without a delta untouched");
        assert_eq!(g.hsl.green.h, 10.0);
    }

    #[test]
    fn voice_mode_defaults_and_names() {
        let a: ClipAudio = serde_json::from_str(r#"{"gainDb":1}"#).unwrap();
        assert_eq!(a.voice, VoiceMode::Original);
        let a: ClipAudio = serde_json::from_str(r#"{"voice":"background"}"#).unwrap();
        assert_eq!(a.voice, VoiceMode::Background);
        let a: ClipAudio = serde_json::from_str(r#"{"voice":"karaoke"}"#).unwrap();
        assert_eq!(a.voice, VoiceMode::Original, "unknown modes load as original");
        let v = serde_json::to_value(ClipAudio::default()).unwrap();
        assert_eq!(v, json!({ "gainDb": 0.0, "normalize": true, "muted": false, "voice": "original" }));
        // stems are omitted until the asset has been separated
        let asset = serde_json::to_value(Asset::default()).unwrap();
        assert!(asset.get("stems").is_none());
    }

    #[test]
    fn partial_documents_load_with_defaults() {
        let p: Project = serde_json::from_str(r#"{"id":"p","name":"n","tracks":[{"id":"t","kind":"video","clips":[{"id":"c","assetId":"a","trackId":"t","inMs":0,"outMs":1000}]}]}"#).unwrap();
        assert_eq!(p.version, 1);
        assert_eq!(p.fps, 24.0);
        assert_eq!((p.width, p.height), (1920, 1080));
        let c = &p.tracks[0].clips[0];
        assert_eq!(c.speed.preset, SpeedPreset::Normal);
        assert_eq!(c.transform.scale.static_value, 1.0);
        assert_eq!(c.transform.opacity.static_value, 1.0);
        assert!(c.color.is_neutral());
        assert!(c.audio.normalize);
        assert!(c.mask.is_none() && c.reframe.is_none() && c.freeze_frame.is_none());
        assert_eq!(c.blend_mode, BlendMode::Normal);
        assert_eq!(c.color.curves.master, vec![[0.0, 0.0], [1.0, 1.0]]);
        assert_eq!(c.color.lut_intensity, 1.0);
    }

    #[test]
    fn pipeline_events_parse() {
        let e: PipelineEvent = serde_json::from_str(r#"{"event":"progress","stage":"shots","clip":"Clip_01.mp4","pct":0.25,"message":"detecting"}"#).unwrap();
        assert!(matches!(e, PipelineEvent::Progress { stage: PipelineStage::Shots, pct, .. } if (pct - 0.25).abs() < 1e-9));
        let e: PipelineEvent = serde_json::from_str(r#"{"event":"log","level":"warn","message":"x"}"#).unwrap();
        assert!(matches!(e, PipelineEvent::Log { level: LogLevel::Warn, .. }));
        let e: PipelineEvent = serde_json::from_str(r#"{"event":"result","path":"C:/out/analysis.json"}"#).unwrap();
        assert!(matches!(e, PipelineEvent::Result { .. }));
        // unknown stage is tolerated
        let e: PipelineEvent = serde_json::from_str(r#"{"event":"progress","stage":"warmup","pct":0}"#).unwrap();
        assert!(matches!(e, PipelineEvent::Progress { stage: PipelineStage::Other, .. }));
        let v = serde_json::to_value(PipelineEvent::Log { level: LogLevel::Error, message: "m".into() }).unwrap();
        assert_eq!(v, json!({"event":"log","level":"error","message":"m"}));
    }

    #[test]
    fn pipeline_options_defaults_and_names() {
        let o: PipelineOptions = serde_json::from_str(r#"{"prompts":["raccoon"],"detector":"grounded_sam2","shotDetector":"transnetv2"}"#).unwrap();
        assert_eq!(o.detector, Detector::GroundedSam2);
        assert_eq!(o.shot_detector, ShotDetector::Transnetv2);
        assert_eq!(o.target_duration_ms, 165_000.0);
        assert_eq!(o.target_lufs, -14.0);
        assert_eq!(o.similarity_threshold, 0.85);
        assert_eq!(o.smoothing, Smoothing::Ema);
        let v = serde_json::to_value(PipelineOptions::default()).unwrap();
        assert_eq!(v["detector"], "hybrid");
        assert_eq!(v["shotDetector"], "auto");
        assert_eq!(v["smoothingAlpha"], 0.15);
    }

    #[test]
    fn analysis_result_round_trip() {
        let json = json!({
            "version": 1, "generatedAt": "2026-09-24T20:00:00Z",
            "clips": [{
                "path": "C:/a.mp4",
                "asset": { "id": "ast_1", "path": "C:/a.mp4", "name": "a.mp4", "kind": "video", "durationMs": 1000, "width": 1920, "height": 1080, "fps": 24, "hasAudio": true },
                "shots": [{ "index": 0, "startFrame": 0, "endFrame": 119, "startMs": 0, "endMs": 4958.3, "confidence": 0.93, "method": "transnetv2", "cast": ["chr_raccoon"] },
                          { "index": 1, "startFrame": 120, "endFrame": 200, "startMs": 5000, "endMs": 8333.3, "confidence": 0.9, "method": "pyscenedetect" }],
                "duplicates": [{ "shotIndex": 0, "frame": 40, "timeMs": 1666,
                    "primary": { "trackId": 1, "label": "raccoon in hoodie", "bbox": [0,0,10,10], "score": 0.9 },
                    "duplicate": { "trackId": 2, "label": "raccoon in hoodie", "bbox": [5,5,15,15], "score": 0.8 },
                    "similarity": 0.94, "character": "chr_raccoon", "characterName": "Raccoon" }],
                "reframe": [{ "shotIndex": 0, "track": { "sourceWidth": 1920, "sourceHeight": 1080, "keyframes": [] } }],
                "audio": { "integratedLufs": -23.1, "truePeakDb": -1.2, "recommendedGainDb": 9.0, "beats": [100.0, 600.0], "tempoBpm": 120.0 },
                "transitions": [{ "fromShot": 0, "toShot": 1, "flowMagnitude": 3.2, "smoothness": 0.71, "suggestion": "dissolve" }]
            }],
            "timeline": { "version": 1, "id": "p", "name": "auto", "fps": 24, "width": 1920, "height": 1080, "assets": [], "tracks": [], "beatMarkers": [] }
        });
        let r: AnalysisResult = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(r.clips[0].transitions[0].suggestion, TransitionSuggestion::Dissolve);
        assert_eq!(r.clips[0].audio.as_ref().unwrap().tempo_bpm, Some(120.0));
        assert_eq!(r.clips[0].shots[0].cast.as_deref(), Some(&["chr_raccoon".to_string()][..]));
        assert_eq!(r.clips[0].shots[1].cast, None);
        assert_eq!(r.clips[0].duplicates[0].character_name.as_deref(), Some("Raccoon"));
        assert_eq!(norm(serde_json::to_value(&r).unwrap()), norm(json));
    }
}
