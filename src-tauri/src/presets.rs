//! The universal adjust preset: one set of slider and HSL values applied on top of every clip's
//! grade in every project (toggleable per project). Stored in `<repo>/presets/universal-adjust.json`
//! so the desktop app, the headless `cappycat-cli` and new projects all share it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};


use crate::model::{AdjustValues, HslAdjustments, HslOffset, UniversalAdjust};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct UniversalPresetFile {
    pub version: u32,
    pub name: String,
    /// New projects (and headless renders of projects that don't say otherwise) start with the
    /// preset switched on.
    pub enabled_by_default: bool,
    pub values: AdjustValues,
}

fn hsl(h: f64, s: f64, l: f64) -> HslOffset {
    HslOffset { h, s, l }
}

impl Default for UniversalPresetFile {
    /// The user's house look (2026-09-25).
    /// Adjust: sharpness 40, brilliance 6, highlights / contrast / exposure 5, temperature -10,
    /// tint 10, saturation 10.
    /// HSL (hue, saturation, brightness): red -24/17/0, orange 16/20/0, yellow -8/17/0,
    /// green -33/50/0, cyan -12/26/0, blue 0/17/-6, purple -23/33/-10, magenta -22/27/0.
    fn default() -> Self {
        Self {
            version: 1,
            name: "Universal adjust".into(),
            enabled_by_default: true,
            values: AdjustValues {
                sharpness: Some(40.0),
                brilliance: Some(6.0),
                highlights: Some(5.0),
                contrast: Some(5.0),
                exposure: Some(5.0),
                temperature: Some(-10.0),
                tint: Some(10.0),
                saturation: Some(10.0),
                hsl: Some(HslAdjustments {
                    red: hsl(-24.0, 17.0, 0.0),
                    orange: hsl(16.0, 20.0, 0.0),
                    yellow: hsl(-8.0, 17.0, 0.0),
                    green: hsl(-33.0, 50.0, 0.0),
                    cyan: hsl(-12.0, 26.0, 0.0),
                    blue: hsl(0.0, 17.0, -6.0),
                    purple: hsl(-23.0, 33.0, -10.0),
                    magenta: hsl(-22.0, 27.0, 0.0),
                }),
                ..Default::default()
            },
        }
    }
}

impl UniversalPresetFile {
    pub fn to_adjust(&self) -> UniversalAdjust {
        UniversalAdjust { enabled: self.enabled_by_default, name: self.name.clone(), values: self.values.clone() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UniversalPresetInfo {
    pub path: String,
    pub preset: UniversalPresetFile,
}

/// `<repo>/presets/universal-adjust.json` in a checkout, `Documents\Cappycat\Presets\…` when
/// installed (seeded from the bundled default; see [`crate::paths`]).
pub fn preset_path() -> Option<PathBuf> {
    Some(crate::paths::resolver().universal_preset())
}

pub fn preset_path_in(repo: &Path) -> PathBuf {
    repo.join("presets").join("universal-adjust.json")
}

/// Load the preset, writing the default file first if it doesn't exist yet.
pub fn load_or_create_at(path: &Path) -> Result<UniversalPresetFile, String> {
    if !path.is_file() {
        let preset = UniversalPresetFile::default();
        save_at(path, &preset)?;
        return Ok(preset);
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} is not a valid preset: {e}", path.display()))
}

pub fn save_at(path: &Path, preset: &UniversalPresetFile) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(preset).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("cannot write {}: {e}", path.display()))
}

pub fn load() -> Result<UniversalPresetInfo, String> {
    let path = preset_path().ok_or("cannot locate the Cappycat project folder")?;
    let preset = load_or_create_at(&path)?;
    Ok(UniversalPresetInfo { path: path.to_string_lossy().into_owned(), preset })
}

pub fn save(preset: &UniversalPresetFile) -> Result<UniversalPresetInfo, String> {
    let path = preset_path().ok_or("cannot locate the Cappycat project folder")?;
    save_at(&path, preset)?;
    Ok(UniversalPresetInfo { path: path.to_string_lossy().into_owned(), preset: preset.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{effective_grade, ColorGrade};

    #[test]
    fn default_file_is_created_with_the_house_values_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("cappycat_preset_{}", uuid::Uuid::new_v4().simple()));
        let path = preset_path_in(&dir);
        let p = load_or_create_at(&path).unwrap();
        assert!(path.is_file());
        assert_eq!(p.values.sharpness, Some(40.0));
        assert_eq!(p.values.brilliance, Some(6.0));
        assert_eq!(p.values.temperature, Some(-10.0));
        let h = p.values.hsl.as_ref().unwrap();
        assert_eq!((h.green.h, h.green.s, h.green.l), (-33.0, 50.0, 0.0));
        assert_eq!((h.purple.h, h.purple.s, h.purple.l), (-23.0, 33.0, -10.0));
        assert!(p.enabled_by_default);
        let mut edited = p.clone();
        edited.values.saturation = Some(12.0);
        save_at(&path, &edited).unwrap();
        assert_eq!(load_or_create_at(&path).unwrap(), edited);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn effective_grade_adds_clamps_and_respects_the_toggle() {
        // CapCut scale: adjust sliders clamp at +-50 (sharpness 0..50), HSL at +-100
        let mut clip = ColorGrade { saturation: 45.0, contrast: -5.0, sharpness: 30.0, ..Default::default() };
        clip.hsl.green.s = 60.0;
        let mut u = UniversalPresetFile::default().to_adjust();
        let g = effective_grade(&clip, Some(&u));
        assert_eq!(g.saturation, 50.0); // 45 + 10 clamped
        assert_eq!(g.contrast, 0.0); // -5 + 5
        assert_eq!(g.sharpness, 50.0); // 30 + 40 clamped
        assert_eq!(g.brilliance, 6.0);
        assert_eq!(g.temperature, -10.0);
        assert_eq!(g.hsl.green.s, 100.0); // 60 + 50 clamped
        assert_eq!(g.hsl.green.h, -33.0);
        assert_eq!(g.hsl.blue.l, -6.0);
        u.enabled = false;
        assert_eq!(effective_grade(&clip, Some(&u)), clip);
        assert_eq!(effective_grade(&clip, None), clip);
    }
}
