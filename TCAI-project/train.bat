@echo off
setlocal

cd /d "%~dp0"

set "DATA_PATH=data\turtle_dataset_finetune_format.json"
set "OUTPUT_DIR=outputs\llama31-turtle-lora"
set "PYTHON_CMD=py"

if exist ".conda312\python.exe" (
    set "PYTHON_CMD=.conda312\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
)

if not exist "%DATA_PATH%" (
    echo Dataset not found: %DATA_PATH%
    exit /b 1
)

echo Starting Unsloth training...
echo Dataset: %DATA_PATH%
echo Output : %OUTPUT_DIR%
echo Python : %PYTHON_CMD%

"%PYTHON_CMD%" train.py --data_path "%DATA_PATH%" --output_dir "%OUTPUT_DIR%" %*

endlocal
