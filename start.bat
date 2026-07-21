@echo off
title RAG System Process Orchestrator

echo Checking for Redis connection on 127.0.0.1:6379...
powershell -Command "try { $socket = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 6379); $socket.Close(); Write-Host 'Redis connection verified.' } catch { Write-Error 'Redis is not running on port 6379. Please ensure Memurai/Redis service is running.' }"

echo Starting Celery background worker...
start "Celery Worker" cmd /k ".\venv\Scripts\celery -A tasks worker --loglevel=info -P solo"

echo Starting FastAPI Backend API Server (Port 8000)...
start "FastAPI API Server" cmd /k ".\venv\Scripts\python -m uvicorn api:app --host 127.0.0.1 --port 8000"

echo Starting Streamlit UI Frontend App (Port 8501)...
start "Streamlit Frontend" cmd /k ".\venv\Scripts\streamlit run app.py --server.port 8501 --server.fileWatcherType none"

echo Starting Watchdog Folder Monitor...
start "Watchdog Directory Watcher" cmd /k ".\venv\Scripts\python watcher.py"

echo.
echo All services launched successfully!
echo -----------------------------------------------------------------
echo - Celery log: See the "Celery Worker" CMD window
echo - FastAPI uvicorn log: See the "FastAPI API Server" CMD window
echo - Streamlit log: See the "Streamlit Frontend" CMD window
echo - Watcher log: See the "Watchdog Directory Watcher" CMD window
echo -----------------------------------------------------------------
echo.
pause
