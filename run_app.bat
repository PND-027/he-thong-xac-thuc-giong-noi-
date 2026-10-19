@echo off
title VoiceID - He thong Xac thuc Giong noi
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   Dang khoi dong VoiceID Web App...
echo   Dia chi web: http://127.0.0.1:7860
echo ========================================================
".venv\Scripts\python.exe" -X utf8 app.py
pause
