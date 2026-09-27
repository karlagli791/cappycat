//! Every location the native core uses, resolved in one place (`docs/FEATURES_V2.md` §9).
//!
//! Resolution order for each location: **1.** an environment override, **2.** the repository
//! layout (dev mode, unchanged from before v2), **3.** the installed layout.
//!
//! **Layout.** The app runs from the *repo* when the executable's folder or one of its four
//! parents is a checkout root — a folder with `pipeline/cappycat_pipeline/__init__.py` **and**
//! `src-tauri/Cargo.toml`. An executable with a bundled `pipeline/` next to it (and no checkout
//! above it) is *installed*, whatever the working directory; otherwise the working directory and
//! its parent are tried, and failing that the layout is *installed*. `CAPPYCAT_LAYOUT=installed`
//! forces the installed layout (tests, trying it from a checkout).
//!
//! | location | override | repo layout | installed layout |
//! |---|---|---|---|
//! | app data home | `CAPPYCAT_HOME` | `%LOCALAPPDATA%\Cappycat` | `%LOCALAPPDATA%\Cappycat` |
//! | documents home | `CAPPYCAT_DOCUMENTS` (else `<CAPPYCAT_HOME>\Documents` when `CAPPYCAT_HOME` is set) | `Documents\Cappycat` | `Documents\Cappycat` |
//! | pipeline | `CAPPYCAT_PIPELINE_DIR` | `<repo>\pipeline` | `<resources>\pipeline` |
//! | resources | `CAPPYCAT_RESOURCES` | – | the executable's folder (Tauri's resource dir on Windows) |
//! | cache | `CAPPYCAT_CACHE_DIR` | `<home>\cache` | `<home>\cache` |
//! | logs / autosave | – | `<home>\logs`, `<home>\autosave` | same |
//! | managed Python env | – | `<home>\python` | `<home>\python` |
//! | ffmpeg (downloaded) | `CAPPYCAT_FFMPEG_DIR` wins in `ffmpeg.rs` | `<home>\ffmpeg\bin` | same |
//! | uv | – | `<home>\bin\uv.exe` | same |
//! | models | `CAPPYCAT_MODELS_DIR` | `<repo>\pipeline\models` (the pipeline's default) | `<home>\models` |
//! | clips (default scan folder) | – | `<repo>\clips` or the newest `<repo>\*clip*` folder | `<documents>\Clips` |
//! | projects / exports | – | `<repo>\projects`, `<repo>\exports` | `<documents>\Projects`, `<documents>\Exports` |
//! | characters | `CAPPYCAT_CHARACTERS` (manifest path) | `<repo>\characters` | `<documents>\Characters` (seeded) |
//! | universal preset | – | `<repo>\presets\universal-adjust.json` | `<documents>\Presets\universal-adjust.json` (seeded) |
//!
//! On Windows `%LOCALAPPDATA%\Cappycat` is the same folder the pre-v2 builds used
//! (`%LOCALAPPDATA%\cappycat`, the file system is case-insensitive), so caches and logs carry over.
//!
//! **Seeding** ([`seed_documents`], run at start-up in the installed layout): `Characters` and
//! `Presets` are copied from the bundled resources when the folder does not exist yet (never
//! overwriting anything).
//!
//! Python children get the matching environment ([`python_env`]): `CAPPYCAT_MODELS_DIR`,
//! `CAPPYCAT_CHARACTERS`, `CAPPYCAT_STEMS_DIR`, `CAPPYCAT_FFMPEG_DIR` and a `PYTHONPYCACHEPREFIX`
//! outside the install folder, wherever they differ from the pipeline's own defaults.

use std::path::{Path, PathBuf};

/// Which layout the app runs in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    /// A source checkout (`<root>` contains `pipeline/` and `src-tauri/`).
    Repo(PathBuf),
    /// An installed app.
    Installed,
}

/// Process state the resolution depends on (a snapshot, so tests can fake it).
#[derive(Debug, Clone, Default)]
pub struct Resolver {
    pub env: std::collections::HashMap<String, String>,
    pub cwd: Option<PathBuf>,
    pub exe_dir: Option<PathBuf>,
    pub local_app_data: Option<PathBuf>,
    pub documents: Option<PathBuf>,
}

const ENV_KEYS: &[&str] = &[
    "CAPPYCAT_HOME",
    "CAPPYCAT_DOCUMENTS",
    "CAPPYCAT_PIPELINE_DIR",
    "CAPPYCAT_RESOURCES",
    "CAPPYCAT_CACHE_DIR",
    "CAPPYCAT_MODELS_DIR",
    "CAPPYCAT_CHARACTERS",
    "CAPPYCAT_LAYOUT",
    "CAPPYCAT_STEMS_DIR",
];

/// Is `dir` a pipeline folder (`cappycat_pipeline/__init__.py` inside)?
pub fn is_pipeline_dir(dir: &Path) -> bool {
    dir.join("cappycat_pipeline").join("__init__.py").is_file()
}

/// Is `dir` the root of a Cappycat checkout?
pub fn is_repo_root(dir: &Path) -> bool {
    is_pipeline_dir(&dir.join("pipeline")) && dir.join("src-tauri").join("Cargo.toml").is_file()
}

fn canon(p: PathBuf) -> PathBuf {
    dunce::canonicalize(&p).unwrap_or(p)
}

impl Resolver {
    /// The current process's state.
    pub fn current() -> Self {
        let env = ENV_KEYS
            .iter()
            .filter_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()).map(|v| (k.to_string(), v)))
            .collect();
        Self {
            env,
            cwd: std::env::current_dir().ok(),
            exe_dir: std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)),
            local_app_data: dirs::data_local_dir().or_else(dirs::cache_dir),
            documents: dirs::document_dir().or_else(|| dirs::home_dir().map(|h| h.join("Documents"))),
        }
    }

    fn var(&self, k: &str) -> Option<PathBuf> {
        self.env.get(k).map(|v| PathBuf::from(v.trim()))
    }

    /// The checkout root, if the app runs from one (and the layout is not forced to installed).
    pub fn repo_root(&self) -> Option<PathBuf> {
        if self.env.get("CAPPYCAT_LAYOUT").map(|v| v.eq_ignore_ascii_case("installed")).unwrap_or(false) {
            return None;
        }
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(p) = self.var("CAPPYCAT_PIPELINE_DIR") {
            if let Some(parent) = p.parent() {
                candidates.push(parent.to_path_buf());
            }
        }
        // the executable's own location wins: a build inside a checkout (target\debug, …) is dev
        // mode, an exe with a bundled pipeline next to it is installed — whatever the cwd is
        if let Some(dir) = &self.exe_dir {
            let mut up = Some(dir.as_path());
            for _ in 0..5 {
                let Some(d) = up else { break };
                candidates.push(d.to_path_buf());
                up = d.parent();
            }
            if let Some(found) = candidates.iter().find(|c| is_repo_root(c)) {
                return Some(canon(found.clone()));
            }
            if is_pipeline_dir(&dir.join("pipeline")) {
                return None;
            }
        }
        if let Some(cwd) = &self.cwd {
            candidates.push(cwd.clone());
            if let Some(p) = cwd.parent() {
                candidates.push(p.to_path_buf());
            }
        }
        candidates.into_iter().find(|c| is_repo_root(c)).map(canon)
    }

    pub fn layout(&self) -> Layout {
        match self.repo_root() {
            Some(r) => Layout::Repo(r),
            None => Layout::Installed,
        }
    }

    pub fn is_installed(&self) -> bool {
        self.layout() == Layout::Installed
    }

    /// `%LOCALAPPDATA%\Cappycat` (or `CAPPYCAT_HOME`).
    pub fn home(&self) -> PathBuf {
        if let Some(h) = self.var("CAPPYCAT_HOME") {
            return h;
        }
        self.local_app_data.clone().unwrap_or_else(std::env::temp_dir).join("Cappycat")
    }

    /// `Documents\Cappycat` (or `CAPPYCAT_DOCUMENTS`, or `<CAPPYCAT_HOME>\Documents`).
    pub fn documents_home(&self) -> PathBuf {
        if let Some(d) = self.var("CAPPYCAT_DOCUMENTS") {
            return d;
        }
        if let Some(h) = self.var("CAPPYCAT_HOME") {
            return h.join("Documents");
        }
        self.documents.clone().unwrap_or_else(|| self.home().join("Documents")).join("Cappycat")
    }

    /// Bundled resources (installed layout): `CAPPYCAT_RESOURCES` or the executable's folder.
    pub fn resources_dir(&self) -> Option<PathBuf> {
        self.var("CAPPYCAT_RESOURCES").or_else(|| self.exe_dir.clone())
    }

    pub fn pipeline_dir(&self) -> Option<PathBuf> {
        if let Some(p) = self.var("CAPPYCAT_PIPELINE_DIR") {
            if is_pipeline_dir(&p) {
                return Some(canon(p));
            }
        }
        if let Some(r) = self.repo_root() {
            return Some(r.join("pipeline"));
        }
        self.resources_dir().map(|r| r.join("pipeline")).filter(|p| is_pipeline_dir(p)).map(canon)
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.var("CAPPYCAT_CACHE_DIR").unwrap_or_else(|| self.home().join("cache"))
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.home().join("logs")
    }

    pub fn autosave_dir(&self) -> PathBuf {
        self.home().join("autosave")
    }

    /// The managed virtual env `setup_ai` creates.
    pub fn python_env_dir(&self) -> PathBuf {
        self.home().join("python")
    }

    pub fn managed_python(&self) -> PathBuf {
        if cfg!(windows) {
            self.python_env_dir().join("Scripts").join("python.exe")
        } else {
            self.python_env_dir().join("bin").join("python")
        }
    }

    /// Where `setup_ai` extracts ffmpeg (`<home>\ffmpeg`, binaries in `bin\`).
    pub fn ffmpeg_dir(&self) -> PathBuf {
        self.home().join("ffmpeg")
    }

    /// Where `setup_ai` puts `uv.exe`.
    pub fn bin_dir(&self) -> PathBuf {
        self.home().join("bin")
    }

    /// The models folder: `CAPPYCAT_MODELS_DIR`, else `<pipeline>\models` in the repo, else
    /// `<home>\models`.
    pub fn models_dir(&self) -> PathBuf {
        if let Some(m) = self.var("CAPPYCAT_MODELS_DIR") {
            return m;
        }
        match (self.repo_root(), self.pipeline_dir()) {
            (Some(_), Some(p)) => p.join("models"),
            _ => self.home().join("models"),
        }
    }

    pub fn characters_dir(&self) -> PathBuf {
        if let Some(m) = self.var("CAPPYCAT_CHARACTERS") {
            if let Some(d) = m.parent() {
                return d.to_path_buf();
            }
        }
        match self.repo_root() {
            Some(r) => r.join("characters"),
            None => self.documents_home().join("Characters"),
        }
    }

    pub fn characters_manifest(&self) -> PathBuf {
        self.var("CAPPYCAT_CHARACTERS").unwrap_or_else(|| self.characters_dir().join("characters.json"))
    }

    pub fn presets_dir(&self) -> PathBuf {
        match self.repo_root() {
            Some(r) => r.join("presets"),
            None => self.documents_home().join("Presets"),
        }
    }

    pub fn universal_preset(&self) -> PathBuf {
        self.presets_dir().join("universal-adjust.json")
    }

    pub fn clips_dir(&self) -> PathBuf {
        match self.repo_root() {
            Some(r) => crate::clips::discover_clips_dir(&r),
            None => self.documents_home().join("Clips"),
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        match self.repo_root() {
            Some(r) => r.join("projects"),
            None => self.documents_home().join("Projects"),
        }
    }

    pub fn exports_dir(&self) -> PathBuf {
        match self.repo_root() {
            Some(r) => r.join("exports"),
            None => self.documents_home().join("Exports"),
        }
    }

    /// Environment for `python -m cappycat_pipeline …` children (see the module docs). Only
    /// variables whose value differs from the pipeline's own default are set, so dev mode keeps
    /// behaving as before.
    pub fn python_env(&self, ffmpeg_dir: Option<&Path>) -> Vec<(String, String)> {
        let s = |p: &Path| p.to_string_lossy().into_owned();
        let mut v = Vec::new();
        let installed = self.is_installed();
        let home_override = self.env.contains_key("CAPPYCAT_HOME");
        if installed || self.env.contains_key("CAPPYCAT_MODELS_DIR") {
            v.push(("CAPPYCAT_MODELS_DIR".into(), s(&self.models_dir())));
        }
        if installed || self.env.contains_key("CAPPYCAT_CHARACTERS") {
            v.push(("CAPPYCAT_CHARACTERS".into(), s(&self.characters_manifest())));
        }
        if (installed || home_override || self.env.contains_key("CAPPYCAT_CACHE_DIR")) && !self.env.contains_key("CAPPYCAT_STEMS_DIR") {
            v.push(("CAPPYCAT_STEMS_DIR".into(), s(&self.cache_dir().join("stems"))));
        }
        if installed {
            // keep __pycache__ out of the install folder (the uninstaller removes only its files)
            v.push(("PYTHONPYCACHEPREFIX".into(), s(&self.cache_dir().join("pycache"))));
        }
        if let Some(d) = ffmpeg_dir {
            v.push(("CAPPYCAT_FFMPEG_DIR".into(), s(d)));
        }
        v
    }
}

/* -------------------------------------------------- process-wide shortcuts */

pub fn resolver() -> Resolver {
    Resolver::current()
}

pub fn layout() -> Layout {
    resolver().layout()
}

pub fn is_installed() -> bool {
    resolver().is_installed()
}

pub fn home() -> PathBuf {
    resolver().home()
}

pub fn cache_dir() -> PathBuf {
    resolver().cache_dir()
}

pub fn logs_dir() -> PathBuf {
    resolver().logs_dir()
}

pub fn pipeline_dir() -> Option<PathBuf> {
    resolver().pipeline_dir()
}

pub fn repo_root() -> Option<PathBuf> {
    resolver().repo_root()
}

/// Environment for python children (see [`Resolver::python_env`]); the ffmpeg folder is the one
/// the native core found, so the pipeline uses the same binaries.
pub fn python_env() -> Vec<(String, String)> {
    let ff = crate::ffmpeg::find_binaries().ok().and_then(|b| b.ffmpeg.parent().map(Path::to_path_buf));
    let ff = ff.filter(|_| std::env::var_os("CAPPYCAT_FFMPEG_DIR").is_none());
    resolver().python_env(ff.as_deref())
}

/* ------------------------------------------------------------------ seeding */

fn copy_tree(src: &Path, dst: &Path, filter: &dyn Fn(&Path) -> bool, copied: &mut Vec<String>) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let p = e.path();
        let name = e.file_name();
        let target = dst.join(&name);
        if p.is_dir() {
            if name.to_string_lossy().starts_with('.') {
                continue; // `.cache` etc. are never bundled / seeded
            }
            copy_tree(&p, &target, filter, copied)?;
        } else if filter(&p) && !target.exists() {
            std::fs::copy(&p, &target)?;
            copied.push(target.to_string_lossy().into_owned());
        }
    }
    Ok(())
}

/// Installed layout: create `Documents\Cappycat\{Clips, Projects, Exports}` and seed
/// `Characters` / `Presets` from the bundled resources when those folders do not exist yet.
/// Returns the files copied. Does nothing in the repo layout.
pub fn seed_documents_with(r: &Resolver) -> Result<Vec<String>, String> {
    if !r.is_installed() {
        return Ok(Vec::new());
    }
    let docs = r.documents_home();
    for d in ["Clips", "Projects", "Exports"] {
        let p = docs.join(d);
        std::fs::create_dir_all(&p).map_err(|e| format!("cannot create {}: {e}", p.display()))?;
    }
    let mut copied = Vec::new();
    let Some(res) = r.resources_dir() else { return Ok(copied) };
    type Seed<'a> = (&'a str, PathBuf, &'a dyn Fn(&Path) -> bool);
    let seeds: [Seed; 2] = [
        ("characters", r.characters_dir(), &|p: &Path| {
            matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("json" | "webp" | "md" | "png" | "jpg"))
        }),
        ("presets", r.presets_dir(), &|p: &Path| p.extension().map(|e| e == "json").unwrap_or(false)),
    ];
    for (name, target, filter) in seeds {
        let src = res.join(name);
        if !src.is_dir() || target.exists() {
            continue;
        }
        copy_tree(&src, &target, filter, &mut copied).map_err(|e| format!("cannot seed {}: {e}", target.display()))?;
    }
    Ok(copied)
}

pub fn seed_documents() -> Result<Vec<String>, String> {
    seed_documents_with(&resolver())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cappycat-paths-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    fn fake_repo(root: &Path) {
        touch(&root.join("pipeline/cappycat_pipeline/__init__.py"));
        touch(&root.join("src-tauri/Cargo.toml"));
    }

    fn fake_install(dir: &Path) {
        touch(&dir.join("pipeline/cappycat_pipeline/__init__.py"));
        touch(&dir.join("characters/characters.json"));
        touch(&dir.join("characters/bunny_ref.webp"));
        touch(&dir.join("characters/README.md"));
        touch(&dir.join("characters/.cache/embeddings.npz"));
        touch(&dir.join("presets/universal-adjust.json"));
    }

    #[test]
    fn repo_layout_is_found_from_cwd_or_exe() {
        let root = tmp("repo");
        fake_repo(&root);
        let r = Resolver { cwd: Some(root.join("src-tauri")), local_app_data: Some(root.join("lad")), documents: Some(root.join("docs")), ..Default::default() };
        let rr = canon(root.clone());
        assert_eq!(r.layout(), Layout::Repo(rr.clone()));
        assert_eq!(r.pipeline_dir(), Some(rr.join("pipeline")));
        assert_eq!(r.models_dir(), rr.join("pipeline").join("models"));
        assert_eq!(r.universal_preset(), rr.join("presets").join("universal-adjust.json"));
        assert_eq!(r.characters_manifest(), rr.join("characters").join("characters.json"));
        assert_eq!(r.home(), root.join("lad").join("Cappycat"));
        assert_eq!(r.cache_dir(), root.join("lad").join("Cappycat").join("cache"));
        // dev mode: python gets no overrides (the pipeline's own defaults apply)
        assert!(r.python_env(None).is_empty());
        // from the exe of target/release
        let r2 = Resolver { exe_dir: Some(root.join("src-tauri/target/release")), ..Default::default() };
        assert_eq!(r2.repo_root(), Some(rr.clone()));
        // forced installed
        let mut r3 = r.clone();
        r3.env.insert("CAPPYCAT_LAYOUT".into(), "installed".into());
        assert_eq!(r3.layout(), Layout::Installed);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn installed_layout_and_overrides() {
        let base = tmp("inst");
        let app = base.join("Programs/Cappycat");
        fake_install(&app);
        let r = Resolver {
            exe_dir: Some(app.clone()),
            cwd: Some(base.join("somewhere")),
            local_app_data: Some(base.join("lad")),
            documents: Some(base.join("Documents")),
            ..Default::default()
        };
        assert!(r.is_installed(), "a bundled pipeline next to the exe is not a checkout");
        // started with a cwd inside a checkout: still installed (the exe's location wins)
        let repo = base.join("checkout");
        fake_repo(&repo);
        let r_cwd = Resolver { cwd: Some(repo.join("src-tauri")), ..r.clone() };
        assert!(r_cwd.is_installed(), "cwd inside a checkout must not switch an installed app to dev mode");
        // but a dev build inside the checkout is dev mode whatever the cwd
        let r_dev = Resolver { exe_dir: Some(repo.join("src-tauri/target/debug")), cwd: Some(base.clone()), ..r.clone() };
        assert_eq!(r_dev.repo_root(), Some(canon(repo.clone())));
        assert_eq!(r.pipeline_dir(), Some(canon(app.join("pipeline"))));
        let home = base.join("lad").join("Cappycat");
        assert_eq!(r.home(), home);
        assert_eq!(r.models_dir(), home.join("models"));
        assert_eq!(r.managed_python(), home.join("python").join("Scripts").join("python.exe"));
        assert_eq!(r.ffmpeg_dir(), home.join("ffmpeg"));
        let docs = base.join("Documents").join("Cappycat");
        assert_eq!(r.clips_dir(), docs.join("Clips"));
        assert_eq!(r.universal_preset(), docs.join("Presets").join("universal-adjust.json"));
        assert_eq!(r.characters_manifest(), docs.join("Characters").join("characters.json"));
        let env: std::collections::HashMap<String, String> = r.python_env(Some(Path::new("C:/ff/bin"))).into_iter().collect();
        assert_eq!(env["CAPPYCAT_MODELS_DIR"], home.join("models").to_string_lossy());
        assert_eq!(env["CAPPYCAT_CHARACTERS"], docs.join("Characters").join("characters.json").to_string_lossy());
        assert_eq!(env["CAPPYCAT_STEMS_DIR"], home.join("cache").join("stems").to_string_lossy());
        assert_eq!(env["CAPPYCAT_FFMPEG_DIR"], "C:/ff/bin");
        assert!(env.contains_key("PYTHONPYCACHEPREFIX"));

        // seeding copies characters (json / webp / md, not .cache) and presets once
        let copied = seed_documents_with(&r).unwrap();
        assert_eq!(copied.len(), 4, "{copied:?}");
        assert!(docs.join("Characters/bunny_ref.webp").is_file());
        assert!(docs.join("Characters/README.md").is_file());
        assert!(!docs.join("Characters/.cache").exists());
        assert!(docs.join("Presets/universal-adjust.json").is_file());
        for d in ["Clips", "Projects", "Exports"] {
            assert!(docs.join(d).is_dir());
        }
        // a user edit survives: nothing is copied again
        std::fs::write(docs.join("Characters/characters.json"), b"edited").unwrap();
        assert!(seed_documents_with(&r).unwrap().is_empty());
        assert_eq!(std::fs::read(docs.join("Characters/characters.json")).unwrap(), b"edited");

        // CAPPYCAT_HOME moves app data and (without CAPPYCAT_DOCUMENTS) the documents
        let mut r2 = r.clone();
        r2.env.insert("CAPPYCAT_HOME".into(), base.join("h").to_string_lossy().into_owned());
        assert_eq!(r2.home(), base.join("h"));
        assert_eq!(r2.documents_home(), base.join("h").join("Documents"));
        assert_eq!(r2.cache_dir(), base.join("h").join("cache"));
        r2.env.insert("CAPPYCAT_DOCUMENTS".into(), base.join("d").to_string_lossy().into_owned());
        assert_eq!(r2.clips_dir(), base.join("d").join("Clips"));
        r2.env.insert("CAPPYCAT_MODELS_DIR".into(), "M:/models".into());
        assert_eq!(r2.models_dir(), PathBuf::from("M:/models"));
        r2.env.insert("CAPPYCAT_PIPELINE_DIR".into(), base.join("nope").to_string_lossy().into_owned());
        assert_eq!(r2.pipeline_dir(), Some(canon(app.join("pipeline"))), "an invalid override falls through");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn this_checkout_runs_in_the_repo_layout() {
        // `cargo test` runs in src-tauri: the tests must see the repo (dev mode unchanged)
        let r = Resolver::current();
        if r.env.contains_key("CAPPYCAT_LAYOUT") {
            return;
        }
        assert!(matches!(r.layout(), Layout::Repo(_)), "{:?}", r.layout());
        assert!(r.pipeline_dir().map(|p| is_pipeline_dir(&p)).unwrap_or(false));
    }
}
