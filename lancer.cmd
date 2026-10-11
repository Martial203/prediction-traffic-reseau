@echo off
rem Demarre l'API (port 8000) et le tableau de bord Angular (port 4200), puis ouvre le navigateur.
cd /d "%~dp0"

if not exist backend\artifacts\metrics.json (
    echo Les modeles ne sont pas entraines : lancez d'abord installer.cmd
    pause
    exit /b 1
)

start "API trafic 4G" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python -m uvicorn app.api:app --port 8000"
start "Tableau de bord trafic 4G" cmd /k "cd /d "%~dp0frontend" && npm start"

echo Demarrage en cours, le navigateur va s'ouvrir...
timeout /t 20 /nobreak >nul
start "" http://localhost:4200
