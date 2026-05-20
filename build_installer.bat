@echo off
REM ============================================================
REM  build_installer.bat  —  Build rag_installer.exe with Nuitka
REM
REM  The installer is stdlib-only (no venv needed at runtime)
REM  so users can run it on a fresh machine.
REM  Output: dist_nuitka\rag_installer.exe
REM ============================================================

setlocal

set PYTHON=.venv\Scripts\python.exe
set OUT_DIR=dist_nuitka

echo [build] Building rag_installer.exe ...
%PYTHON% -m nuitka ^
    --standalone ^
    --onefile ^
    --enable-plugin=tk-inter ^
    --windows-console-mode=disable ^
    --windows-icon-from-ico=installer\assets\icon.ico ^
    --output-filename=rag_installer.exe ^
    --output-dir=%OUT_DIR% ^
    --include-data-file=db\schema.sql=db\schema.sql ^
    --include-data-file=configs\runtime.yaml=configs\runtime.yaml ^
    --include-data-file=installer\assets\icon.ico=installer\assets\icon.ico ^
    --noinclude-pytest-mode=nofollow ^
    installer\installer.py

if errorlevel 1 (
    echo [build] FAILED.
    exit /b 1
)
echo [build] Done: %OUT_DIR%\rag_installer.exe
endlocal
