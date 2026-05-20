@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Building Personal AI (RAG) — PyInstaller bundle
echo ============================================================
echo.

REM ── 1. Build with PyInstaller ────────────────────────────────────────────────
echo [1/4] Running PyInstaller...
.venv\Scripts\pyinstaller rag.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller failed.  See output above.
    pause
    exit /b 1
)
echo.

REM ── 2. Copy user-editable configs beside the exe ─────────────────────────────
echo [2/4] Copying default configs...
xcopy /E /I /Y configs dist\rag\configs
if errorlevel 1 (
    echo ERROR: Failed to copy configs.
    pause
    exit /b 1
)
echo.

REM ── 3. Create empty data directories (first-run directories) ─────────────────
echo [3/4] Creating data directories...
mkdir dist\rag\data\db           2>NUL
mkdir dist\rag\data\diagnostics  2>NUL
mkdir dist\rag\data\feedback     2>NUL
mkdir dist\rag\data\metadata     2>NUL
mkdir dist\rag\data\chunks       2>NUL
mkdir dist\rag\data\index        2>NUL
echo.

REM ── 4. Summary ───────────────────────────────────────────────────────────────
echo [4/4] Done!
echo.
echo   Distribution:  dist\rag\
echo   Executable:    dist\rag\rag.exe
echo   Data folder:   dist\rag\data\        (grows with ingested documents)
echo   Config folder: dist\rag\configs\     (edit these to change server settings)
echo.
echo To run:  dist\rag\rag.exe
echo          (opens http://localhost:8000 in your browser automatically)
echo.
echo To distribute:  zip up dist\rag\ and send it.
echo                 The recipient just extracts and runs rag.exe.
echo.
pause
endlocal
