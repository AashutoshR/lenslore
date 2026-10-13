@echo off
rem Double-click to start LensLore. Keep this window open while you use the app.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app.py
pause
