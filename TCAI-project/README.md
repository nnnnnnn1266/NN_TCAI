# TCAI 專案

這個 repo 目前包含兩個方向：

- FastAPI 問答示範系統
- 使用 Unsloth 對 LLaMA 3.1 8B 進行領域問答微調的訓練流程

本次新增的微調流程使用：

- 模型：`unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- 方法：LoRA + `SFTTrainer`
- 量化：4-bit
- 精度：自動切換 `bf16` / `fp16`
- 輸出：LoRA adapter

## 資料格式

訓練資料採 instruction tuning JSON 格式，每筆資料需包含三個欄位：

```json
[
  {
    "instruction": "你是一位烏龜照護專家，請回答問題。",
    "input": "為什麼烏龜需要曬背？",
    "output": "烏龜曬背有助於調節體溫、乾燥甲殼，並促進 UVB 相關代謝。"
  }
]
```

預設範例檔位於 [`data/train_example.json`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/data/train_example.json)。

你的正式資料集目前也已放入 repo：

- [`data/turtle_dataset_finetune_format.json`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/data/turtle_dataset_finetune_format.json)

## 安裝

建議先建立虛擬環境，再安裝套件：

```bash
pip install -r requirements.txt
```

## 訓練指令

在 `TCAI-project` 目錄下執行：

```bash
python train.py ^
  --data_path data/turtle_dataset_finetune_format.json ^
  --output_dir outputs/llama31-turtle-lora
```

如果你的環境是用 Python Launcher：

```bash
py train.py ^
  --data_path data/turtle_dataset_finetune_format.json ^
  --output_dir outputs/llama31-turtle-lora
```

常用可調參數範例：

```bash
python train.py ^
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

訓練完成後，LoRA adapter 會儲存在你指定的 `output_dir`。

## 一鍵訓練

如果你是在 Windows 環境，也可以直接使用：

```bat
train.bat
```

這會自動使用：

- 資料集：`data\turtle_dataset_finetune_format.json`
- 輸出目錄：`outputs\llama31-turtle-lora`

如果要覆蓋額外參數，也可以這樣執行：

```bat
train.bat --num_train_epochs 5 --batch_size 1
```

## 推論指令

使用已訓練好的 LoRA adapter 進行推論：

```bash
python inference.py ^
  --adapter_path outputs/llama31-turtle-lora ^
  --instruction "你是一位烏龜照護專家，請根據專業知識回答。" ^
  --input_text "巴西龜不吃東西可能有哪些原因？"
```

或使用 `py`：

```bash
py inference.py ^
  --adapter_path outputs/llama31-turtle-lora ^
  --instruction "你是一位烏龜照護專家，請根據專業知識回答。" ^
  --input_text "巴西龜不吃東西可能有哪些原因？"
```

## 主要檔案

- [`train.py`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/train.py)：資料前處理、LoRA 設定與 SFT 訓練流程
- [`inference.py`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/inference.py)：載入 LoRA adapter 並產生回答
- [`data/train_example.json`](c:/Users/Teacher/Desktop/NN_TCAI/NN_TCAI/TCAI-project/data/train_example.json)：instruction tuning 範例資料

## FastAPI 示範系統

如果你也要啟動現有的 FastAPI 範例：

```bash
uvicorn app.main:app --reload
```

然後開啟 `http://127.0.0.1:8000`。

如果 `outputs/llama31-turtle-lora` 存在且環境已安裝 Unsloth，`/ask` 會自動載入 LoRA adapter 進行推論；否則會自動退回 mock 回答。

你也可以用環境變數指定 adapter 路徑：

```bash
set TCAI_ADAPTER_PATH=outputs\llama31-turtle-lora
uvicorn app.main:app --reload
```

`GET /health` 會回傳目前後端使用的是 `lora` 還是 `mock` 模式。

## 注意事項

- 第一次訓練會下載基礎模型與 tokenizer，因此需要可連網環境與足夠磁碟空間。
- `unsloth/Meta-Llama-3.1-8B-bnb-4bit` 需遵守對應模型授權。
- 4-bit QLoRA 雖然比全參數微調省資源，但仍需要可用 GPU。依 Unsloth 文件，8B 模型的 4-bit 微調通常至少需要約 6GB VRAM，實務上仍會隨 batch size、context length 等設定而上升。

參考資料：

- Unsloth 文件：https://docs.unsloth.ai/
- Unsloth Requirements：https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements
- Unsloth LLaMA 3.1 4-bit 模型頁：https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
