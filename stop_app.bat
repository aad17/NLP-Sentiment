@echo off
REM Stop Sentiment Analysis Application
REM This script stops all components of the Fire Brigade Sentiment Analysis application
REM Windows version

setlocal enabledelayedexpansion

call :print_header "Stopping Fire Brigade Sentiment Analysis Application..."
call :print_yellow "Note: All trained models (including the expanded sentiment model) will be preserved."

REM Check if process file exists
if exist ".app_processes.bat" (
    call :print_blue "Stopping application components using stored commands..."
    call .app_processes.bat
    del .app_processes.bat
) else (
    call :print_yellow "No process file found. Attempting to find and stop processes..."
    
    REM Stop Mock API
    call :print_blue "Stopping Mock API processes..."
    taskkill /f /fi "WINDOWTITLE eq mock_api.py*" 2>nul
    taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq *mock_api.py*" 2>nul
    call :print_green "Attempted to stop Mock API processes."
    
    REM Stop Model API
    call :print_blue "Stopping Model API processes..."
    taskkill /f /fi "WINDOWTITLE eq model_api.py*" 2>nul
    taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq *model_api.py*" 2>nul
    call :print_green "Attempted to stop Model API processes."
    
    REM Stop Streamlit Dashboard
    call :print_blue "Stopping Streamlit Dashboard processes..."
    taskkill /f /fi "WINDOWTITLE eq streamlit run app\dashboard\dashboard.py*" 2>nul
    call :print_green "Attempted to stop Streamlit Dashboard processes."
    
    REM Stop Social Media Batch Processor
    call :print_blue "Stopping Social Media Batch Processor..."
    taskkill /f /fi "WINDOWTITLE eq data_processor.py*" 2>nul
    taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq *data_processor.py*" 2>nul
    call :print_green "Attempted to stop Social Media Batch Processor."
    
    REM Stop Social Media Auth UI
    call :print_blue "Stopping Social Media Auth UI..."
    taskkill /f /fi "WINDOWTITLE eq streamlit run app\social_media\social_media_auth.py*" 2>nul
    call :print_green "Attempted to stop Social Media Auth UI."
    
    REM Stop Social Media Data UI
    call :print_blue "Stopping Social Media Data UI..."
    taskkill /f /fi "WINDOWTITLE eq streamlit run app\social_media\social_media_data.py*" 2>nul
    call :print_green "Attempted to stop Social Media Data UI."
    
    REM Stop CSV Upload UI
    call :print_blue "Stopping CSV Upload UI..."
    taskkill /f /fi "WINDOWTITLE eq streamlit run app\social_media\csv_data_upload.py*" 2>nul
    call :print_green "Attempted to stop CSV Upload UI."
    
    REM Stop all remaining Streamlit processes
    call :print_blue "Stopping any remaining Streamlit processes..."
    taskkill /f /fi "IMAGENAME eq streamlit.exe" 2>nul
    call :print_green "Attempted to stop remaining Streamlit processes."
)

REM Also check for any remaining Python processes related to our app (optional failsafe)
call :print_blue "Checking for any remaining application processes..."

REM Dashboard
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fi "windowtitle eq *dashboard.py*" /fo list ^| find "PID:"') do (
    call :print_yellow "Found remaining dashboard process with PID: %%p. Stopping it..."
    taskkill /f /pid %%p 2>nul
)

REM Model API
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fi "windowtitle eq *model_api.py*" /fo list ^| find "PID:"') do (
    call :print_yellow "Found remaining model API process with PID: %%p. Stopping it..."
    taskkill /f /pid %%p 2>nul
)

REM Mock API
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fi "windowtitle eq *mock_api.py*" /fo list ^| find "PID:"') do (
    call :print_yellow "Found remaining mock API process with PID: %%p. Stopping it..."
    taskkill /f /pid %%p 2>nul
)

REM Social Media Processor
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fi "windowtitle eq *data_processor.py*" /fo list ^| find "PID:"') do (
    call :print_yellow "Found remaining Social Media Processor with PID: %%p. Stopping it..."
    taskkill /f /pid %%p 2>nul
)

REM Social Media Auth
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fi "windowtitle eq *social_media_auth.py*" /fo list ^| find "PID:"') do (
    call :print_yellow "Found remaining Social Media Auth process with PID: %%p. Stopping it..."
    taskkill /f /pid %%p 2>nul
)

REM Social Media Data
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fi "windowtitle eq *social_media_data.py*" /fo list ^| find "PID:"') do (
    call :print_yellow "Found remaining Social Media Data process with PID: %%p. Stopping it..."
    taskkill /f /pid %%p 2>nul
)

REM CSV Upload
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fi "windowtitle eq *csv_data_upload.py*" /fo list ^| find "PID:"') do (
    call :print_yellow "Found remaining CSV Upload process with PID: %%p. Stopping it..."
    taskkill /f /pid %%p 2>nul
)

call :print_header "Fire Brigade Sentiment Analysis Application stopped."
call :print_blue "All sentiment models (including the expanded model) are preserved."
call :print_blue "Run start_app.bat to restart the application."
call :print_header "Shutdown complete."

goto :eof

REM ======= Functions =======

:print_header
echo.
echo ==========================================
echo %~1
echo ==========================================
echo.
goto :eof

:print_green
echo [SUCCESS] %~1
goto :eof

:print_red
echo [ERROR] %~1
goto :eof

:print_yellow
echo [WARNING] %~1
goto :eof

:print_blue
echo [INFO] %~1
goto :eof 