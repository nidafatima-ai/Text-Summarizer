@echo off
echo Starting T5 Text Summarizer Server...
echo.

:: Open the browser automatically after 3 seconds
start "" "http://127.0.0.1:8000"

:: Start the Uvicorn server
uvicorn main:app --reload

pause