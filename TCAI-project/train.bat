@echo off
setlocal

cd /d "%~dp0"

set "DATA_PATH=data\turtle_dataset_finetune_format.json"
set "OUTPUT_DIR=outputs\llama31-turtle-lora"

if not exist "%DATA_PATH%" (
    echo Dataset not found: %DATA_PATH%
    exit /b 1
)

echo Starting Unsloth training...
echo Dataset: %DATA_PATH%
echo Output : %OUTPUT_DIR%

py train.py --data_path "%DATA_PATH%" --output_dir "%OUTPUT_DIR%" %*

endlocal
