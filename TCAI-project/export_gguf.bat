@echo off
setlocal

cd /d "%~dp0"

set "ADAPTER_PATH=outputs\llama31-turtle-lora"
set "OUTPUT_DIR=exports\llama31-turtle-gguf"
set "QUANTIZATION_METHOD=q4_k_m"
set "PYTHON_CMD=py"

if exist ".conda312\python.exe" (
    set "PYTHON_CMD=.conda312\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
)

if not exist "%ADAPTER_PATH%" (
    echo Adapter not found: %ADAPTER_PATH%
    exit /b 1
)

echo Starting GGUF export...
echo Adapter : %ADAPTER_PATH%
echo Output  : %OUTPUT_DIR%
echo Quant   : %QUANTIZATION_METHOD%
echo Python  : %PYTHON_CMD%

"%PYTHON_CMD%" export_gguf.py --adapter_path "%ADAPTER_PATH%" --output_dir "%OUTPUT_DIR%" --quantization_method "%QUANTIZATION_METHOD%" %*

endlocal
