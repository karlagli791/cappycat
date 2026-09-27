//! Adobe / Resolve `.cube` 3D LUT parser and trilinear sampler.
//!
//! Mirrors `src/engine/color/lut.ts`: red varies fastest in the file, the
//! lattice is sampled with plain trilinear interpolation over `c * (size-1)`
//! (identical to the preview's `texture(u_lut, c*(s-1)/s + 0.5/s)` with
//! LINEAR filtering). `DOMAIN_MIN/MAX` are parsed but — like the preview —
//! not applied. Values are kept in f32 (the preview uploads them as RGB8, a
//! difference of at most 1/255).

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Largest `.cube` file `load_cube` reads (a 128³ LUT is ~70 MB of text).
const MAX_CUBE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Lut3D {
    pub size: usize,
    /// `size^3` RGB triplets, index `(b * size + g) * size + r`.
    pub data: Vec<[f32; 3]>,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    pub title: Option<String>,
}

fn parse_triplet(parts: &[&str]) -> Option<[f32; 3]> {
    if parts.len() < 3 {
        return None;
    }
    let r = parts[0].parse::<f32>().ok()?;
    let g = parts[1].parse::<f32>().ok()?;
    let b = parts[2].parse::<f32>().ok()?;
    (r.is_finite() && g.is_finite() && b.is_finite()).then_some([r, g, b])
}

/// Parse the text of a `.cube` file (a leading UTF-8 byte-order mark is ignored).
pub fn parse_cube(text: &str) -> Result<Lut3D, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut size = 0usize;
    let mut title = None;
    let mut domain_min = [0.0f32; 3];
    let mut domain_max = [1.0f32; 3];
    let mut data: Vec<[f32; 3]> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("TITLE") {
            title = Some(rest.trim().trim_matches('"').to_string());
            continue;
        }
        if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
            size = rest.trim().parse().map_err(|_| format!("invalid LUT_3D_SIZE: {line}"))?;
            continue;
        }
        if line.starts_with("LUT_1D_SIZE") {
            return Err("1D LUTs are not supported; expected LUT_3D_SIZE".into());
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if line.starts_with("DOMAIN_MIN") {
            domain_min = parse_triplet(&parts[1..]).ok_or("invalid DOMAIN_MIN")?;
            continue;
        }
        if line.starts_with("DOMAIN_MAX") {
            domain_max = parse_triplet(&parts[1..]).ok_or("invalid DOMAIN_MAX")?;
            continue;
        }
        if let Some(t) = parse_triplet(&parts) {
            data.push(t);
        }
        // Unknown keywords (LUT_3D_INPUT_RANGE, …) are ignored like the preview does.
    }
    if size == 0 {
        return Err("Missing LUT_3D_SIZE".into());
    }
    if !(2..=128).contains(&size) {
        return Err(format!("Unsupported LUT size {size}"));
    }
    let expected = size * size * size;
    if data.len() != expected {
        return Err(format!("Expected {} values for a {size}^3 LUT, got {}", expected * 3, data.len() * 3));
    }
    Ok(Lut3D { size, data, domain_min, domain_max, title })
}

/// Read and parse a `.cube` file. The bytes are decoded as lossy UTF-8 (LUTs
/// written by Windows tools in a legacy code page only differ in their TITLE).
pub fn load_cube(path: &Path) -> Result<Lut3D, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("cannot read LUT {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("LUT {} is not a file", path.display()));
    }
    if meta.len() > MAX_CUBE_BYTES {
        return Err(format!("LUT {} is too large ({} MB)", path.display(), meta.len() / (1024 * 1024)));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read LUT {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_cube(&text).map_err(|e| format!("invalid LUT {}: {e}", path.display()))
}

/// A parsed LUT as returned to the preview by the `load_lut` command:
/// `data` holds `size³` RGB triplets flattened (`length = size³ · 3`), red varying fastest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LutPayload {
    pub size: usize,
    pub data: Vec<f32>,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl From<&Lut3D> for LutPayload {
    fn from(l: &Lut3D) -> Self {
        Self {
            size: l.size,
            data: l.data.iter().flat_map(|v| v.iter().copied()).collect(),
            domain_min: l.domain_min,
            domain_max: l.domain_max,
            title: l.title.clone(),
        }
    }
}

impl Lut3D {
    /// Identity lattice of the given size.
    pub fn identity(size: usize) -> Self {
        let size = size.max(2);
        let n = (size - 1) as f32;
        let mut data = Vec::with_capacity(size * size * size);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    data.push([r as f32 / n, g as f32 / n, b as f32 / n]);
                }
            }
        }
        Self { size, data, domain_min: [0.0; 3], domain_max: [1.0; 3], title: None }
    }

    /// Serialise as `.cube` text (used by tests and fixtures).
    pub fn to_cube_text(&self) -> String {
        let mut s = String::new();
        if let Some(t) = &self.title {
            s.push_str(&format!("TITLE \"{t}\"\n"));
        }
        s.push_str(&format!("LUT_3D_SIZE {}\n", self.size));
        for v in &self.data {
            s.push_str(&format!("{:.6} {:.6} {:.6}\n", v[0], v[1], v[2]));
        }
        s
    }

    /// Trilinear sample for an input colour in 0..1 (clamped).
    #[inline]
    pub fn sample(&self, c: [f32; 3]) -> [f32; 3] {
        let s = self.size;
        let n = (s - 1) as f32;
        let fx = (c[0] * n).clamp(0.0, n);
        let fy = (c[1] * n).clamp(0.0, n);
        let fz = (c[2] * n).clamp(0.0, n);
        let (x0, y0, z0) = (fx as usize, fy as usize, fz as usize); // floor (non-negative)
        let (tx, ty, tz) = (fx - x0 as f32, fy - y0 as f32, fz - z0 as f32);
        // neighbour strides (0 on the last lattice plane = clamp)
        let dx = usize::from(x0 + 1 < s);
        let dy = if y0 + 1 < s { s } else { 0 };
        let dz = if z0 + 1 < s { s * s } else { 0 };
        let base = (z0 * s + y0) * s + x0;
        let d = &self.data;
        let (p000, p100, p010, p110) = (d[base], d[base + dx], d[base + dy], d[base + dy + dx]);
        let (p001, p101, p011, p111) = (d[base + dz], d[base + dz + dx], d[base + dz + dy], d[base + dz + dy + dx]);
        let mut out = [0.0; 3];
        for i in 0..3 {
            let c00 = p000[i] + (p100[i] - p000[i]) * tx;
            let c10 = p010[i] + (p110[i] - p010[i]) * tx;
            let c01 = p001[i] + (p101[i] - p001[i]) * tx;
            let c11 = p011[i] + (p111[i] - p011[i]) * tx;
            let c0 = c00 + (c10 - c00) * ty;
            let c1 = c01 + (c11 - c01) * ty;
            out[i] = c0 + (c1 - c0) * tz;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    #[test]
    fn identity_lut_is_identity() {
        let lut = Lut3D::identity(17);
        for c in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.25, 0.5, 0.75], [0.123, 0.987, 0.456]] {
            assert!(close3(lut.sample(c), c, 1e-5), "{c:?} -> {:?}", lut.sample(c));
        }
    }

    #[test]
    fn parses_cube_text_round_trip() {
        let mut lut = Lut3D::identity(3);
        lut.title = Some("warm".into());
        for v in lut.data.iter_mut() {
            v[0] = (v[0] * 0.9 + 0.1).min(1.0);
        }
        let text = format!("# comment\n{}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n", lut.to_cube_text());
        let parsed = parse_cube(&text).unwrap();
        assert_eq!(parsed.size, 3);
        assert_eq!(parsed.title.as_deref(), Some("warm"));
        assert!(close3(parsed.sample([0.0, 0.0, 0.0]), [0.1, 0.0, 0.0], 1e-5));
        // halfway between lattice points 0 and 1 on red: (0.1 + 0.55) / 2
        assert!(close3(parsed.sample([0.25, 0.0, 0.0]), [0.325, 0.0, 0.0], 1e-5));
    }

    #[test]
    fn bom_and_non_utf8_files_load() {
        let dir = crate::ffmpeg::cache_dir().join("test").join("lut_encodings");
        std::fs::create_dir_all(&dir).unwrap();
        let mut lut = Lut3D::identity(2);
        lut.title = Some("look".into());
        // UTF-8 with a BOM, CRLF line ends
        let with_bom = dir.join("bom.cube");
        let text = format!("\u{feff}{}", lut.to_cube_text().replace('\n', "\r\n"));
        std::fs::write(&with_bom, text.as_bytes()).unwrap();
        let a = load_cube(&with_bom).unwrap();
        assert_eq!((a.size, a.title.as_deref()), (2, Some("look")));
        // a BOM right before LUT_3D_SIZE (no title) must not hide the keyword
        std::fs::write(&with_bom, format!("\u{feff}LUT_3D_SIZE 2\n{}", "0 0 0\n".repeat(8))).unwrap();
        assert_eq!(load_cube(&with_bom).unwrap().size, 2);
        // Windows-1252 title (0xE9 = é) is not valid UTF-8
        let latin = dir.join("latin1.cube");
        let mut bytes = b"TITLE \"caf\xE9\"\nLUT_3D_SIZE 2\n".to_vec();
        bytes.extend("0.5 0.5 0.5\n".repeat(8).as_bytes());
        std::fs::write(&latin, bytes).unwrap();
        let b = load_cube(&latin).unwrap();
        assert_eq!(b.size, 2);
        assert!(b.title.unwrap().starts_with("caf"));
        // clear errors
        let err = load_cube(&dir.join("missing.cube")).unwrap_err();
        assert!(err.contains("cannot read LUT") && err.contains("missing.cube"), "{err}");
        std::fs::write(&latin, "LUT_3D_SIZE 3\n0 0 0\n").unwrap();
        let err = load_cube(&latin).unwrap_err();
        assert!(err.contains("invalid LUT") && err.contains("Expected 81 values"), "{err}");
        // the load_lut payload: flattened, red fastest
        let mut l = Lut3D::identity(2);
        l.data[1] = [0.9, 0.1, 0.2];
        let p = LutPayload::from(&l);
        assert_eq!(p.data.len(), 2 * 2 * 2 * 3);
        assert_eq!(&p.data[3..6], &[0.9, 0.1, 0.2]);
        let v = serde_json::to_value(&p).unwrap();
        assert!(v.get("domainMin").is_some() && v.get("title").is_none());
    }

    #[test]
    fn rejects_bad_files() {
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 0\n").unwrap_err().contains("Expected 24 values"));
        assert!(parse_cube("0 0 0\n").unwrap_err().contains("LUT_3D_SIZE"));
        assert!(parse_cube("LUT_1D_SIZE 16\n").is_err());
        assert!(parse_cube("LUT_3D_SIZE 1\n").is_err());
    }
}
