@echo off
rem Installation complete : environnement Python, entrainement des modeles, dependances Angular.
setlocal
cd /d "%~dp0"

echo.
echo [1/5] Environnement Python (backend\.venv)
if not exist backend\.venv\Scripts\python.exe (
    python -m venv backend\.venv || goto :error
)
backend\.venv\Scripts\python -m pip install --upgrade pip --quiet

echo.
echo [2/5] Dependances Python
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt --retries 10 --timeout 120 || goto :error

echo.
echo [3/5] PyTorch (LSTM) - telechargement volumineux
backend\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --retries 10 --timeout 120
if errorlevel 1 echo [!] PyTorch non installe : le LSTM sera ignore, les autres modeles fonctionnent.

echo.
echo [4/5] Entrainement et evaluation des modeles
pushd backend
.venv\Scripts\python -m app.train
if errorlevel 1 (popd & goto :error)
popd

echo.
echo [5/5] Dependances du frontend Angular
pushd frontend
call npm install --prefer-offline --no-audit --no-fund
if errorlevel 1 (popd & goto :error)
popd

echo.
echo Installation terminee. Lancez maintenant lancer.cmd
pause
exit /b 0

:error
echo.
echo [X] L'installation a echoue a l'etape ci-dessus.
pause
exit /b 1
