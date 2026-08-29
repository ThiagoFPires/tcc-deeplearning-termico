@echo off
title ThermoScan AI - Inicializador do Sistema
cd /d "%~dp0"

echo ============================================================
echo   ThermoScan AI - Mastologia Termica com Deep Learning
echo ============================================================
echo.
echo [1/2] Iniciando o servidor Backend (FastAPI na porta 8000)...
start "ThermoScan API (FastAPI)" cmd /k ".\venv\Scripts\python.exe -m uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000"

echo [2/2] Iniciando o servidor Frontend (Interface Web na porta 3000)...
start "ThermoScan Frontend (Web)" cmd /k ".\venv\Scripts\python.exe -m http.server 3000 --directory interface"

timeout /t 2 >nul
echo.
echo Abrindo a Interface Web no seu navegador padrao...
start http://127.0.0.1:3000/

echo.
echo ============================================================
echo  Sistema online com sucesso!
echo  - Frontend: http://127.0.0.1:3000/
echo  - Documentacao API: http://127.0.0.1:8000/docs
echo ============================================================
pause
