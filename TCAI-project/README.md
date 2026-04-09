# TCAI 專案

這個 repo 目前包含兩個部分：

- 烏龜照護問答的 FastAPI 示範系統
- 使用 Unsloth 對 LLaMA 3.1 8B 進行 instruction tuning 的微調流程

目前程式碼已支援：

- 基礎模型：`unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- 方法：4-bit + LoRA + `SFTTrainer`
- 訓練資料格式：`instruction` / `input` / `output`
- 輸出：LoRA adapter
- API 自動切換：
  若找到已訓練 adapter，`/ask` 會使用 LoRA 模型；否則退回 mock 回答

## 專案結構

```text
TCAI-project/
├─ app/
│  ├─ api/
│  ├─ models/
│  ├─ services/
│  └─ main.py
├─ data/
├─ docs/
├─ export_gguf.py
├─ export_gguf.bat
├─ frontend/
├─ inference.py
├─ train.py
├─ train.bat
├─ requirements.txt
└─ README.md
```

## 重要檔案

- [`train.py`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/train.py)
  使用 Unsloth 載入 4-bit LLaMA 3.1 8B，做資料驗證、prompt 格式化、LoRA 設定與 SFT 訓練
- [`inference.py`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/inference.py)
  載入已訓練的 LoRA adapter，從命令列做推論
- [`export_gguf.py`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/export_gguf.py)
  把已訓練好的 LoRA adapter 匯出成 GGUF
- [`export_gguf.bat`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/export_gguf.bat)
  Windows 一鍵 GGUF 匯出入口
- [`train.bat`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/train.bat)
  Windows 一鍵訓練入口
- [`app/services/inference.py`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/app/services/inference.py)
  FastAPI 用的推論層。會自動檢查 adapter 是否存在，並在 `lora` / `mock` 間切換
- [`app/main.py`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/app/main.py)
  FastAPI 入口，提供 `/`、`/ask`、`/health`
- [`frontend/index.html`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/frontend/index.html)
  簡易前端頁面，可直接輸入問題並查看目前模式

## 資料格式

訓練資料必須是 JSON 陣列，每筆資料都要包含：

- `instruction`
- `input`
- `output`

範例：

```json
[
  {
    "instruction": "You are a turtle care domain expert. Answer with accurate, practical, and concise guidance.",
    "input": "Why do aquatic turtles need a basking area?",
    "output": "Aquatic turtles need a basking area to regulate body temperature, dry their shells, and support healthy metabolism when proper heat and UVB are available."
  }
]
```

目前 repo 內的資料檔：

- [`data/train_example.json`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/data/train_example.json)
- [`data/turtle_dataset_finetune_format.json`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/data/turtle_dataset_finetune_format.json)

## 安裝

在 `TCAI-project` 目錄下安裝套件：

```bash
pip install -r requirements.txt
```

或：

```bash
py -m pip install -r requirements.txt
```

## Windows 新電腦安裝流程

以下流程已在一台 Windows + NVIDIA RTX 5070 Ti 的新電腦上實際跑通。

### 建議環境

- 作業系統：Windows
- Python：建議 `3.12`
- GPU：NVIDIA GPU
- 驅動：`nvidia-smi` 可正常執行

### 1. 建立專案環境

建議使用專案內的 Conda 環境：

```powershell
conda --no-plugins create --solver classic -p .conda312 python=3.12 pip -y
```

### 2. 安裝 PyTorch CUDA 版

先裝 CUDA 版 PyTorch：

```powershell
.conda312\python.exe -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安裝 Unsloth 與專案套件

```powershell
.conda312\python.exe -m pip install unsloth
.conda312\python.exe -m pip install -r requirements.txt
```

### 4. 驗證 GPU / PyTorch / CUDA

```powershell
.conda312\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

預期至少要看到：

- `torch.cuda.is_available()` 為 `True`
- 顯示 NVIDIA 顯卡名稱
- 顯示 CUDA 版本

### 5. 執行訓練

目前 `train.bat` 會優先使用：

- `.conda312\python.exe`
- 否則 `.venv\Scripts\python.exe`
- 否則退回系統 `py`

所以直接執行：

```powershell
.\train.bat
```

### 6. 啟動 FastAPI

```powershell
.conda312\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 7. 測試 API 與首頁

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-WebRequest -Uri http://127.0.0.1:8000/
```

測試問答：

```powershell
$body = @{ question = "Why do aquatic turtles need a basking area?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body $body
```

如果 LoRA 成功載入，`/health` 應會回傳類似：

```json
{
  "status": "ok",
  "mode": "lora",
  "detail": "Loaded adapter from: ...\\outputs\\llama31-turtle-lora"
}
```

## 這次實測結果

在這台新電腦上已完成：

- 建立 `.conda312`
- 安裝 CUDA 版 PyTorch、Unsloth、FastAPI 相依
- 驗證 GPU / CUDA 可用
- 成功執行 `train.bat`
- 成功產出 `outputs/llama31-turtle-lora`
- 成功啟動 FastAPI
- 成功測試 `/health`、首頁 `/`、以及 `/ask`

## 訓練

### Windows 一鍵訓練

```bat
train.bat
```

`train.bat` 預設會使用：

- 資料集：`data\turtle_dataset_finetune_format.json`
- 輸出目錄：`outputs\llama31-turtle-lora`

你也可以加參數覆蓋：

```bat
train.bat --num_train_epochs 5 --batch_size 1
```

### 手動執行訓練

```bash
py train.py --data_path data/turtle_dataset_finetune_format.json --output_dir outputs/llama31-turtle-lora
```

常用參數範例：

```bash
py train.py ^
  --data_path data/turtle_dataset_finetune_format.json ^
  --output_dir outputs/llama31-turtle-lora ^
  --max_seq_length 2048 ^
  --batch_size 2 ^
  --gradient_accumulation_steps 4 ^
  --num_train_epochs 3 ^
  --learning_rate 2e-4 ^
  --lora_r 16 ^
  --lora_alpha 16
```

訓練成功後，LoRA adapter 會存到你指定的 `output_dir`。

## 命令列推論

如果已經有訓練好的 adapter，可以這樣測試：

```bash
py inference.py ^
  --adapter_path outputs/llama31-turtle-lora ^
  --instruction "You are a professional turtle care assistant. Provide a clear and accurate answer." ^
  --input_text "Why did my turtle suddenly stop eating?"
```

## 匯出 GGUF

如果你想把目前訓練好的 LoRA 匯出成可下載的單一 GGUF 模型，可以使用：

```powershell
.conda312\python.exe export_gguf.py ^
  --adapter_path outputs/llama31-turtle-lora ^
  --output_dir exports/llama31-turtle-gguf ^
  --quantization_method q4_k_m
```

Windows 也可以直接一鍵執行：

```powershell
.\export_gguf.bat
```

常用量化格式：

- `q4_k_m`
- `q5_k_m`
- `q8_0`
- `f16`

輸出完成後，GGUF 檔案會在你指定的 `output_dir` 內。

這台機器實測成功的輸出檔案位置為：

```text
exports/llama31-turtle-gguf_gguf/Meta-Llama-3.1-8B.Q4_K_M.gguf
```

注意：

- 匯出的 GGUF 是「完整模型」，檔案會比 LoRA adapter 大很多。
- 匯出過程需要能夠正確載入 base model 與 adapter。
- Windows 首次匯出時，Unsloth 可能會自動準備 `llama.cpp`，因此第一次會比較久。
- 如果匯出時顯示記憶體不足，可以把 `--maximum_memory_usage` 降低，例如：

```powershell
.conda312\python.exe export_gguf.py ^
  --maximum_memory_usage 0.5
```

## FastAPI 與前端測試

啟動後端：

```bash
uvicorn app.main:app --reload
```

開啟：

- 首頁：`http://127.0.0.1:8000`
- 健康檢查：`http://127.0.0.1:8000/health`

目前 API 行為如下：

- `GET /`
  回傳前端頁面
- `POST /ask`
  接收問題並回傳答案
- `GET /health`
  回傳目前後端是 `lora` 還是 `mock`

如果 `outputs/llama31-turtle-lora` 存在，而且環境能成功載入 Unsloth 與 adapter，`/ask` 會用 LoRA 模型回答。

如果 adapter 不存在、Unsloth 未安裝，或模型載入失敗，系統會自動退回 mock 回答，不會讓 API 直接壞掉。

你也可以指定自訂 adapter 路徑：

```bash
set TCAI_ADAPTER_PATH=outputs\llama31-turtle-lora
uvicorn app.main:app --reload
```

可選環境變數：

- `TCAI_ADAPTER_PATH`
- `TCAI_MAX_SEQ_LENGTH`
- `TCAI_SYSTEM_PROMPT`

## 目前已知限制

- Unsloth 訓練需要可用的 GPU 環境。
- 前端目前是英文文案，但功能完整，會顯示目前模式是 `LoRA model` 或 `Mock fallback`。
- `train.py` 目前可以執行，但在有 GPU 的環境中建議優先確認 PyTorch、CUDA、Unsloth 版本是否相容。

## 建議測試順序

1. `git clone` 這個 repo
2. 安裝相依套件與 GPU 版 PyTorch
3. 執行 `train.bat`
4. 訓練完成後啟動 `uvicorn app.main:app --reload`
5. 用 `/health` 確認目前是 `lora`

## GitHub 備註

`.gitignore` 已排除：

- `outputs/`
- `logs/`
- `*.log`
- `__pycache__/`
- `.venv/`
- `.env`

因此你可以先把這個 repo 推到 GitHub，再換到有 GPU 的電腦訓練。
