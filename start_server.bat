@echo off
cd /d "%~dp0"
echo Starting RAG server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
.venv\Scripts\python.exe main.py serve --host 0.0.0.0 --port 8000 --reload
