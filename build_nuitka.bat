@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Building Personal AI (RAG) — Nuitka
echo  Single EXE: GUI launcher + server in one binary
echo ============================================================
echo.
echo  Double-click rag.exe  →  opens the launcher GUI
echo  rag.exe serve         →  headless server mode
echo  rag.exe ingest ...    →  CLI ingest
echo.
echo  First build takes 30-90 minutes. Subsequent builds are faster.
echo.

REM ── 1. Compile with Nuitka ───────────────────────────────────────────────────
echo [1/4] Compiling with Nuitka (this takes a while)...
.venv\Scripts\python -m nuitka ^
    --standalone ^
    --low-memory ^
    --lto=no ^
    --output-dir=dist_nuitka ^
    --output-filename=rag.exe ^
    --windows-console-mode=attach ^
    --follow-imports ^
    --enable-plugin=tk-inter ^
    --include-package=launcher ^
    --include-package=installer ^
    --include-package=server ^
    --include-package=services ^
    --include-package=db ^
    --include-package=llm ^
    --include-package=retrieval ^
    --include-package=pipeline ^
    --include-package=utils ^
    --include-package=extraction ^
    --include-package=api ^
    --include-package=fastapi ^
    --include-package=uvicorn ^
    --include-package=starlette ^
    --include-package=pydantic ^
    --include-package=httpx ^
    --include-package=fitz ^
    --include-package=pdfplumber ^
    --include-package=pdfminer ^
    --include-package=PIL ^
    --include-package=yaml ^
    --include-package=rank_bm25 ^
    --include-package=trafilatura ^
    --include-package=anyio ^
    --include-package=h11 ^
    --include-package=psycopg ^
    --include-package=pgvector ^
    --include-package=multipart ^
    --include-package=torch ^
    --include-package=sentence_transformers ^
    --include-package=transformers ^
    --include-package=marker ^
    --include-package=surya ^
    --include-package=sv_ttk ^
    --include-data-dir=server/static=server/static ^
    --include-data-files=db/schema.sql=db/schema.sql ^
    --include-data-dir=configs=configs ^
    --include-data-files=installer/assets/icon.ico=installer/assets/icon.ico ^
    --noinclude-pytest-mode=nofollow ^
    --noinclude-IPython-mode=nofollow ^
    --enable-plugin=no-qt ^
    --module-parameter=torch-disable-jit=yes ^
    --nofollow-import-to=transformers.commands ^
    --nofollow-import-to=transformers.testing_utils ^
    --nofollow-import-to=transformers.integrations ^
    --nofollow-import-to=transformers.onnx ^
    --nofollow-import-to=transformers.sagemaker ^
    --nofollow-import-to=transformers.trainer ^
    --nofollow-import-to=pymupdf.mupdf ^
    --include-data-files=.venv/Lib/site-packages/pymupdf/mupdf.py=pymupdf/mupdf.py ^
    --nofollow-import-to=sentence_transformers.sparse_encoder ^
    --nofollow-import-to=sentence_transformers.losses ^
    --nofollow-import-to=sentence_transformers.trainer ^
    --nofollow-import-to=sentence_transformers.training_args ^
    --nofollow-import-to=sentence_transformers.fit_mixin ^
    --nofollow-import-to=sentence_transformers.evaluation ^
    --nofollow-import-to=torch.testing ^
    --nofollow-import-to=torch._inductor ^
    --nofollow-import-to=torch._dynamo ^
    --nofollow-import-to=torch.distributed ^
    --nofollow-import-to=torch.onnx ^
    --nofollow-import-to=sympy ^
    --nofollow-import-to=openai ^
    --nofollow-import-to=google.genai ^
    --nofollow-import-to=google.generativeai ^
    --nofollow-import-to=anthropic ^
    --nofollow-import-to=cohere ^
    --nofollow-import-to=litellm ^
    --assume-yes-for-downloads ^
    --windows-icon-from-ico=installer\assets\icon.ico ^
    main.py

if errorlevel 1 (
    echo.
    echo ERROR: Nuitka compilation failed. See output above.
    pause
    exit /b 1
)
echo.

REM ── 2. Create writable data directories ──────────────────────────────────────
echo [2/4] Creating data directories...
mkdir dist_nuitka\main.dist\data\db           2>NUL
mkdir dist_nuitka\main.dist\data\diagnostics  2>NUL
mkdir dist_nuitka\main.dist\data\feedback     2>NUL
mkdir dist_nuitka\main.dist\data\metadata     2>NUL
mkdir dist_nuitka\main.dist\data\chunks       2>NUL
mkdir dist_nuitka\main.dist\data\index        2>NUL
echo.

REM ── 2b. Copy pymupdf native extensions to root (required for mupdf import fallback) ──
echo    Copying pymupdf native extensions to dist root...
copy /Y .venv\Lib\site-packages\pymupdf\_mupdf.pyd        dist_nuitka\main.dist\_mupdf.pyd        >NUL 2>&1
copy /Y .venv\Lib\site-packages\pymupdf\mupdfcpp64.dll    dist_nuitka\main.dist\mupdfcpp64.dll    >NUL 2>&1
copy /Y .venv\Lib\site-packages\pymupdf\mupdf.py          dist_nuitka\main.dist\mupdf.py          >NUL 2>&1
REM  Also keep copies in pymupdf\ subdir for relative-import path
copy /Y .venv\Lib\site-packages\pymupdf\_mupdf.pyd        dist_nuitka\main.dist\pymupdf\_mupdf.pyd        >NUL 2>&1
copy /Y .venv\Lib\site-packages\pymupdf\mupdfcpp64.dll    dist_nuitka\main.dist\pymupdf\mupdfcpp64.dll    >NUL 2>&1
copy /Y .venv\Lib\site-packages\pymupdf\_features.pyd     dist_nuitka\main.dist\pymupdf\_features.pyd     >NUL 2>&1
echo.

REM ── 3. Rename output folder to rag ───────────────────────────────────────────
echo [3/4] Renaming output...
REM Kill any running rag.exe so the old folder can be fully deleted
taskkill /F /IM rag.exe >NUL 2>&1
timeout /T 1 /NOBREAK >NUL
if exist dist_nuitka\rag rmdir /S /Q dist_nuitka\rag 2>NUL
ren dist_nuitka\main.dist rag 2>NUL
REM Check rename actually succeeded (not just that 'rag' exists from failed delete)
if exist dist_nuitka\rag\rag.exe (
    echo   Renamed to dist_nuitka\rag
) else (
    echo   NOTE: Rename blocked (OneDrive sync or file lock^). Output is in dist_nuitka\main.dist\
    echo   You can rename it manually once OneDrive finishes syncing.
    set RAG_DIR=main.dist
    goto summary_main
)
set RAG_DIR=rag
goto summary_rag

:summary_main
echo.
echo [4/4] Done!
echo.
echo   Distribution:  dist_nuitka\main.dist\
echo   Executable:    dist_nuitka\main.dist\rag.exe
echo.
echo To run:  dist_nuitka\main.dist\rag.exe
echo.
echo To distribute:  zip up dist_nuitka\main.dist\ and send it.INFO:     127.0.0.1:57194 - "GET /api/status HTTP/1.1" 200 OK
goto end

:summary_rag

echo.
REM ── 4. Summary ───────────────────────────────────────────────────────────────
echo [4/4] Done!
echo.
echo   Distribution:  dist_nuitka\rag\
echo   Executable:    dist_nuitka\rag\rag.exe
echo.
echo To run:  dist_nuitka\rag\rag.exe
echo          (opens http://localhost:8000 automatically)
echo.
echo To distribute:  zip up dist_nuitka\rag\ and send it.

:end
echo.
pause
endlocal
