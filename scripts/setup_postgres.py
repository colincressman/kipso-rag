#!/usr/bin/env python3
"""
setup_postgres.py

Checks for PostgreSQL 16 + pgvector on Windows.
If either is missing, downloads and installs them silently.
Creates the 'rag' database and enables the vector extension.

Run once before starting the server for the first time.
Safe to re-run — skips any step that's already done.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
PG_VERSION_MAJOR  = 16
PG_VERSION_FULL   = "16.8"
PG_PORT           = 5432
PG_SERVICE_NAME   = f"postgresql-{PG_VERSION_MAJOR}"
PG_SUPERPASSWORD  = "postgres"      # local service account only
RAG_DB            = "rag"

PGVECTOR_VERSION  = "0.8.0"

# VS Build Tools bootstrap URL (stable Microsoft redirect)
VS_BUILDTOOLS_URL = "https://aka.ms/vs/17/release/vs_buildtools.exe"

# EDB silent installer
PG_INSTALLER_URL = (
    f"https://get.enterprisedb.com/postgresql/"
    f"postgresql-{PG_VERSION_FULL}-1-windows-x64.exe"
)

# pgvector pre-built Windows binaries (pg16)
PGVECTOR_ZIP_URL = (
    f"https://github.com/pgvector/pgvector/releases/download/"
    f"v{PGVECTOR_VERSION}/"
    f"pgvector-v{PGVECTOR_VERSION}-pg{PG_VERSION_MAJOR}-windows-x86_64.zip"
)

# Common EDB install roots
_EDB_ROOTS = [
    Path("C:/Program Files/PostgreSQL"),
    Path("C:/Program Files (x86)/PostgreSQL"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[setup_postgres] {msg}", flush=True)


def _run(args: list, *, check: bool = True, capture: bool = False, env=None):
    """Run a subprocess, optionally capturing output."""
    return subprocess.run(
        args,
        check=check,
        capture_output=capture,
        text=True,
        env=env,
    )


def _download(url: str, dest: Path) -> None:
    _log(f"Downloading {url}")
    _log(f"  → {dest}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


# ── Step 1 — Locate PostgreSQL ─────────────────────────────────────────────────

def _find_pg_bin() -> Path | None:
    """Return the PostgreSQL bin directory if found, else None."""
    # Check PATH first
    pg_ctl = shutil.which("pg_ctl")
    if pg_ctl:
        return Path(pg_ctl).parent

    # Check known EDB install locations
    for root in _EDB_ROOTS:
        if not root.exists():
            continue
        # Prefer the exact major version, then newest
        candidate = root / str(PG_VERSION_MAJOR) / "bin"
        if candidate.exists():
            return candidate
        # Fall back to any version dir
        version_dirs = sorted(root.iterdir(), reverse=True)
        for vd in version_dirs:
            bin_dir = vd / "bin"
            if (bin_dir / "pg_ctl.exe").exists():
                return bin_dir

    return None


# ── Step 2 — Install PostgreSQL ────────────────────────────────────────────────

def _install_postgresql() -> Path:
    """Download the EDB installer and run it silently. Returns the bin dir."""
    _log(f"PostgreSQL {PG_VERSION_MAJOR} not found. Installing...")

    # Use a persistent location so cleanup doesn't race with the running installer
    download_dir = Path(tempfile.gettempdir()) / "rag_pg_setup"
    download_dir.mkdir(exist_ok=True)
    installer = download_dir / "pg_installer.exe"

    if not installer.exists():
        _download(PG_INSTALLER_URL, installer)
    else:
        _log(f"Using cached installer at {installer}")

    _log("Running installer (this takes ~2 minutes, a UAC prompt will appear)...")
    install_args = " ".join([
        f'--mode unattended',
        f'--superpassword "{PG_SUPERPASSWORD}"',
        f'--servicename "{PG_SERVICE_NAME}"',
        f'--servicepassword "{PG_SUPERPASSWORD}"',
        f'--serverport {PG_PORT}',
        f'--prefix "C:\\Program Files\\PostgreSQL\\{PG_VERSION_MAJOR}"',
        f'--datadir "C:\\Program Files\\PostgreSQL\\{PG_VERSION_MAJOR}\\data"',
        f'--enable-components server,commandlinetools',
        f'--disable-components pgAdmin,stackbuilder',
    ])
    import ctypes, time
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", str(installer), install_args, None, 1
    )
    if ret <= 32:
        raise RuntimeError(
            f"ShellExecuteW failed (code {ret}). Try running this script as Administrator."
        )

    _log("Installer launched. Waiting for it to complete (up to 5 minutes)...")
    for i in range(60):
        time.sleep(5)
        found = _find_pg_bin()
        if found:
            _log(f"PostgreSQL detected after {(i+1)*5}s.")
            break
        if i % 6 == 5:
            _log(f"  Still waiting... ({(i+1)*5}s)")
    else:
        raise RuntimeError(
            "Installer did not complete within 5 minutes. "
            "Check if it finished manually, then re-run this script."
        )

    bin_dir = _find_pg_bin()
    if not bin_dir:
        raise RuntimeError(
            "Installer finished but pg_ctl not found. "
            "Check C:\\Program Files\\PostgreSQL\\ manually."
        )
    _log(f"PostgreSQL installed at: {bin_dir}")
    return bin_dir


# ── Step 3 — Ensure service is running ────────────────────────────────────────

def _ensure_running(bin_dir: Path) -> None:
    """Start the PostgreSQL service if it's not already running."""
    pg_ctl = bin_dir / "pg_ctl.exe"
    data_dir = bin_dir.parent / "data"

    result = _run(
        ["sc", "query", PG_SERVICE_NAME],
        check=False, capture=True,
    )
    if "RUNNING" in result.stdout:
        _log("PostgreSQL service is already running.")
        return

    _log("Starting PostgreSQL service...")
    # Try sc first (service installed by EDB installer)
    sc_result = _run(["sc", "start", PG_SERVICE_NAME], check=False, capture=True)
    if sc_result.returncode == 0:
        _log("Service started via sc.")
        return

    # Fallback: pg_ctl start (if running without service)
    if data_dir.exists():
        _run([str(pg_ctl), "start", "-D", str(data_dir), "-w"])
        _log("Service started via pg_ctl.")
    else:
        raise RuntimeError(
            f"Cannot start PostgreSQL: service '{PG_SERVICE_NAME}' not found "
            f"and data dir '{data_dir}' does not exist."
        )


# ── Step 4 — Create rag database ───────────────────────────────────────────────

def _ensure_database(bin_dir: Path) -> None:
    """Create the 'rag' database if it doesn't already exist."""
    psql = bin_dir / "psql.exe"
    createdb = bin_dir / "createdb.exe"

    # Check if the DB already exists
    result = _run(
        [str(psql), "-U", "postgres", "-p", str(PG_PORT),
         "-lqt", "--no-password"],
        check=False, capture=True,
        env={**os.environ, "PGPASSWORD": PG_SUPERPASSWORD},
    )
    if f" {RAG_DB} " in result.stdout:
        _log(f"Database '{RAG_DB}' already exists.")
        return

    _log(f"Creating database '{RAG_DB}'...")
    _run(
        [str(createdb), "-U", "postgres", "-p", str(PG_PORT),
         "--no-password", RAG_DB],
        env={**os.environ, "PGPASSWORD": PG_SUPERPASSWORD},
    )
    _log(f"Database '{RAG_DB}' created.")


# ── Step 5 — Install pgvector extension ───────────────────────────────────────

def _find_pg_sharedir(bin_dir: Path) -> tuple[Path, Path]:
    """Return (sharedir/extension, pkglibdir) using pg_config."""
    pg_config = bin_dir / "pg_config.exe"
    sharedir = Path(
        _run([str(pg_config), "--sharedir"], capture=True).stdout.strip()
    ) / "extension"
    pkglibdir = Path(
        _run([str(pg_config), "--pkglibdir"], capture=True).stdout.strip()
    )
    return sharedir, pkglibdir


def _pgvector_installed(bin_dir: Path) -> bool:
    """Return True if vector.control already exists in the extension dir."""
    sharedir, _ = _find_pg_sharedir(bin_dir)
    return (sharedir / "vector.control").exists()


def _find_vcvars() -> Path | None:
    """Locate vcvars64.bat using vswhere.exe."""
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) \
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    result = _run(
        [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
        capture=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    vs_root = Path(result.stdout.strip())
    vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return vcvars if vcvars.exists() else None


def _install_vs_buildtools() -> None:
    """Install VS Build Tools with the VC++ workload, silently."""
    # Try winget first — ships with Windows 10 (2004+) and Windows 11
    winget_ok = _run(["winget", "--version"], check=False, capture=True).returncode == 0
    if winget_ok:
        _log("Installing VS Build Tools via winget (5-15 min, ~2 GB download)...")
        result = _run(
            [
                "winget", "install",
                "--id", "Microsoft.VisualStudio.2022.BuildTools",
                "-e", "--source", "winget", "--accept-package-agreements",
                "--override",
                "--quiet --wait --norestart "
                "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended",
            ],
            check=False, capture=False,
        )
        if result.returncode == 0:
            return
        _log("winget install returned a non-zero code — falling back to direct download.")

    # Direct-download fallback (same installer, same flags)
    download_dir = Path(tempfile.gettempdir()) / "rag_vsbt_setup"
    download_dir.mkdir(exist_ok=True)
    installer = download_dir / "vs_buildtools.exe"
    if not installer.exists():
        _download(VS_BUILDTOOLS_URL, installer)
    else:
        _log(f"Using cached installer at {installer}")

    _log("Installing VS Build Tools (5-15 min)...")
    _run([
        str(installer),
        "--quiet", "--wait", "--norestart",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--includeRecommended",
    ])


def _ensure_msvc() -> Path:
    """Return vcvars64.bat, installing VS Build Tools automatically if needed."""
    vcvars = _find_vcvars()
    if vcvars:
        _log(f"Found MSVC at: {vcvars.parent.parent.parent.parent}")
        return vcvars

    _log("Visual Studio Build Tools (MSVC) not found — installing automatically...")
    _install_vs_buildtools()

    vcvars = _find_vcvars()
    if not vcvars:
        raise RuntimeError(
            "VS Build Tools were installed but vcvars64.bat was not found.\n"
            "This can happen if the installer needs a restart to finish.\n"
            "Please restart your machine and re-run this script."
        )
    _log(f"Found MSVC at: {vcvars.parent.parent.parent.parent}")
    return vcvars


def _install_pgvector(bin_dir: Path) -> None:
    """Build pgvector from source using MSVC and install into PostgreSQL."""
    _log(f"Installing pgvector {PGVECTOR_VERSION} (building from source)...")

    vcvars = _ensure_msvc()

    src_url = f"https://github.com/pgvector/pgvector/archive/refs/tags/v{PGVECTOR_VERSION}.zip"
    download_dir = Path(tempfile.gettempdir()) / "rag_pgvector_setup"
    download_dir.mkdir(exist_ok=True)
    zip_path = download_dir / f"pgvector-{PGVECTOR_VERSION}.zip"

    if not zip_path.exists():
        _download(src_url, zip_path)
    else:
        _log(f"Using cached source at {zip_path}")

    extract_dir = download_dir / f"pgvector-{PGVECTOR_VERSION}"
    if not extract_dir.exists():
        _log("Extracting source...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(download_dir)

    pg_root = bin_dir.parent
    _log(f"Building with PGROOT={pg_root} ...")

    # Write a .bat file — avoids all quoting/escaping issues with paths containing spaces
    bat_path = download_dir / "build_pgvector.bat"
    bat_path.write_text(
        f"@echo off\n"
        f"call \"{vcvars}\"\n"
        f"if errorlevel 1 exit /b 1\n"
        f"cd /d \"{extract_dir}\"\n"
        f"if errorlevel 1 exit /b 1\n"
        f"set \"PGROOT={pg_root}\"\n"
        f"nmake /F Makefile.win\n"
        f"if errorlevel 1 exit /b 1\n"
        f"nmake /F Makefile.win install\n"
        f"if errorlevel 1 exit /b 1\n"
    )
    result = subprocess.run([str(bat_path)], text=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"pgvector build failed (exit {result.returncode}). "
            "Check output above for errors."
        )
    _log("pgvector built and installed.")

def _enable_vector_extension(bin_dir: Path) -> None:
    """Run CREATE EXTENSION IF NOT EXISTS vector in the rag database."""
    psql = bin_dir / "psql.exe"
    _log("Enabling vector extension in 'rag' database...")
    _run(
        [str(psql), "-U", "postgres", "-p", str(PG_PORT),
         "-d", RAG_DB, "--no-password",
         "-c", "CREATE EXTENSION IF NOT EXISTS vector;"],
        env={**os.environ, "PGPASSWORD": PG_SUPERPASSWORD},
    )
    _log("vector extension enabled.")


# ── Step 6 — Run db migrations ─────────────────────────────────────────────────

def _run_migrations() -> None:
    """Call init_db to apply the schema / run any pending migrations."""
    _log("Running schema migrations...")
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from db.client import init_db
    from utils.config import load_config
    cfg = load_config()
    db_dsn = cfg.get("paths", {}).get("db_dsn") or f"postgresql://postgres:{PG_SUPERPASSWORD}@localhost/{RAG_DB}"
    init_db(db_dsn)
    _log("Migrations complete.")


# ── Main ───────────────────────────────────────────────────────────────────────

def _is_admin() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _elevate_and_restart() -> None:
    """Re-launch this script with admin rights via UAC, then exit this process."""
    import ctypes
    script = str(Path(sys.argv[0]).resolve())
    args   = " ".join(f'"{a}"' for a in sys.argv[1:])
    _log("Administrator rights required — requesting elevation (UAC prompt will appear)...")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas",
        sys.executable, f'"{script}" {args}',
        None, 1,
    )
    if ret <= 32:
        raise RuntimeError(
            f"Failed to elevate (ShellExecuteW returned {ret}). "
            "Try right-clicking your terminal and choosing 'Run as Administrator', then re-run."
        )
    sys.exit(0)  # elevated copy takes over


def main() -> None:
    if sys.platform != "win32":
        _log("This script is Windows-only. On Linux/macOS install PostgreSQL via your package manager.")
        sys.exit(1)

    if not _is_admin():
        _elevate_and_restart()

    # 1. Find or install PostgreSQL
    bin_dir = _find_pg_bin()
    if bin_dir:
        _log(f"Found PostgreSQL at: {bin_dir}")
    else:
        bin_dir = _install_postgresql()

    # 2. Ensure service is running
    _ensure_running(bin_dir)

    # 3. Create the rag database
    _ensure_database(bin_dir)

    # 4. Install pgvector if missing
    if _pgvector_installed(bin_dir):
        _log("pgvector already installed.")
    else:
        _install_pgvector(bin_dir)
        _enable_vector_extension(bin_dir)

    # 5. Apply schema / migrations
    _run_migrations()

    _log("")
    _log("✓ PostgreSQL is ready. DSN: postgresql://localhost/rag")


if __name__ == "__main__":
    main()
