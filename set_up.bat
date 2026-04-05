// Archivo set_up.bat para configurar el entorno de desarrollo en Windows. Crea un entorno virtual, instala dependencias y prepara la estructura de carpetas.
// Generado por GPT-5.3 Codex High, Abril 2025

@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo [1/6] Detecting Python...
set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
	py -3.13 -V >nul 2>nul
	if not errorlevel 1 (
		set "PY_CMD=py -3.13"
	) else (
		py -3 -V >nul 2>nul
		if not errorlevel 1 set "PY_CMD=py -3"
	)
)

if not defined PY_CMD (
	where python >nul 2>nul
	if not errorlevel 1 (
		python -V >nul 2>nul
		if not errorlevel 1 set "PY_CMD=python"
	)
)

if not defined PY_CMD (
	echo [ERROR] Python 3 was not found in PATH.
	echo Install Python 3.13 and run this script again.
	exit /b 1
)

echo [2/6] Creating or reusing virtual environment...
if not exist "env_project\Scripts\python.exe" (
	%PY_CMD% -m venv "env_project"
	if errorlevel 1 (
		echo [ERROR] Could not create virtual environment at env_project.
		exit /b 1
	)
) else (
	echo Existing virtual environment found at env_project.
)

call "env_project\Scripts\activate.bat"
if errorlevel 1 (
	echo [ERROR] Could not activate virtual environment.
	exit /b 1
)

echo [3/6] Upgrading pip tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
	echo [ERROR] Failed to upgrade pip tooling.
	exit /b 1
)

echo [4/6] Installing dependencies...
if exist "requirements.txt" (
	set "REQ_TMP=%TEMP%\requirements_financial_risk_%RANDOM%%RANDOM%.txt"

	powershell -NoProfile -ExecutionPolicy Bypass -Command "$content = Get-Content -LiteralPath '.\requirements.txt' -Raw; Set-Content -LiteralPath '%REQ_TMP%' -Value $content -Encoding utf8"

	if errorlevel 1 (
		echo [WARN] Could not normalize requirements encoding. Using original file.
		python -m pip install -r "requirements.txt"
		if errorlevel 1 (
			echo [ERROR] Dependency installation failed.
			exit /b 1
		)
	) else (
		python -m pip install -r "%REQ_TMP%"
		set "PIP_STATUS=%ERRORLEVEL%"
		del /q "%REQ_TMP%" >nul 2>nul
		if not "%PIP_STATUS%"=="0" (
			echo [ERROR] Dependency installation failed.
			exit /b %PIP_STATUS%
		)
	)
) else (
	echo [WARN] requirements.txt was not found. Skipping dependency installation.
)

echo [5/6] Ensuring project directories exist...
for %%D in (
	"outputs"
	"outputs\artefactos"
	"outputs\modelo_heuristico"
	"outputs\reporte_training_evaluation"
	"src\data"
) do (
	if not exist "%%~D" mkdir "%%~D"
)

if not exist "src\data\logs_produccion.csv" (
	type nul > "src\data\logs_produccion.csv"
)

echo [6/6] Writing .env with local absolute paths...
set "ROOT_DIR=%CD%"

> ".env" echo DATA_FOLDER=%ROOT_DIR%\src\data
>> ".env" echo OUTPUTS=%ROOT_DIR%\outputs
>> ".env" echo ARTIFACTS=%ROOT_DIR%\outputs\artefactos
>> ".env" echo REPORT=%ROOT_DIR%\outputs\reporte_training_evaluation
>> ".env" echo PROJECT_GCP=pro-cientificos-pba

echo.
echo Setup completed successfully.
echo Next steps:
echo   1) Activate venv: env_project\Scripts\activate
echo   2) Go to src:     cd src
echo   3) Build data:    python ft_engineering.py
echo   4) Train model:   python model_training_evaluation.py

exit /b 0