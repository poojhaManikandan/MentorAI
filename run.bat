@echo off
title MentorAI Runner
echo ===================================================
echo   MentorAI - Classroom Intelligence System
echo ===================================================
echo.

set "STREAMLIT_PATH=%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\streamlit.exe"

if exist "%STREAMLIT_PATH%" (
    echo [OK] Found Streamlit at:
    echo      %STREAMLIT_PATH%
    echo.
    echo Launching Streamlit server...
    "%STREAMLIT_PATH%" run app.py
) else (
    echo [WARN] Streamlit not found at default package location.
    echo Trying system path resolution...
    echo.
    streamlit run app.py
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to start Streamlit. Please make sure Python and Streamlit are installed.
)
pause
