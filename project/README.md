# CAIL2018 罪名預測（Charge Prediction）

本專案使用 CAIL2018 練習賽資料，建立**罪名（accusation）多標籤分類**模型。流程為：

`fact`（案情描述） → Tokenizer → 預訓練語言模型（Encoder）→ Attention pooling → 全連接分類頭 → 罪名多標籤輸出

---

## 資料集

- **資料來源**：CAIL2018（中國裁判文書案件推理資料集 / 練習賽 split）
- **輸入欄位**：`fact`（字串，案情描述）
- **輸出欄位**：`meta.accusation`（罪名列表，`list[str]`）
- **檔案位置**：`project/data/`
  - `data_train.json` / `data_valid.json` / `data_test.json`
  - 本 repo 也支援（且 GitHub 上預設提供）對應的壓縮檔：`*.json.gz`

### 檔案格式（NDJSON）

本資料為 **NDJSON**：每一行是一個 JSON 物件（不是整檔一個 JSON array）。

單筆資料結構範例（示意）：

```json
{
  "fact": "...",
  "meta": {
    "accusation": ["故意伤害", "..."],
    "relevant_articles": [234, ...],
    "term_of_imprisonment": { "...": "..." },
    "criminals": ["..."],
    "punish_of_money": 0
  }
}
```

---

## 任務目標

給定 `fact`（案情文字），預測可能成立的罪名 `accusation`。

- **任務類型**：多標籤分類（multi-label classification）
- **損失函數**：`BCEWithLogitsLoss`
- **輸出**：每個罪名一個 logit，推論時經 sigmoid 得到機率，再以閾值（預設 0.5）決定是否輸出該罪名

---

## 模型方法

### 預訓練骨干（Encoder）

使用 Hugging Face `transformers` 載入 encoder：

- 預設（較省記憶體，適合筆電/Mac）：`hfl/chinese-legal-electra-small-discriminator`
- 可在 `project/config.py` 修改 `PRETRAINED_MODEL_NAME` 切換其他 backbone（例如 large 版）

> 本模型採用 `AutoModel`，因此可支援 BERT/ELECTRA/RoBERTa 類 encoder，只要其輸出包含 `last_hidden_state`。

### Attention pooling（token → sentence）

從 encoder 取得 `last_hidden_state`，形狀為 `[B, L, d]`，其中：

- `B`：batch size
- `L`：序列長度（`MAX_LEN`）
- `d`：hidden size（由 `model.config.hidden_size` 決定，不寫死 768）

使用 Bahdanau-style additive attention 對 token 維度做加權平均：

\[
u_i = v^\top \tanh(W h_i + b),\quad
\alpha = \text{softmax}(u),\quad
z = \sum_i \alpha_i h_i
\]

並以 `attention_mask` 對 padding token 做 mask，避免 padding 影響權重。

### 分類頭（FC Head）

句向量 `z ∈ R^d` → 兩層全連接：

- `Linear(d → 512) → ReLU → Dropout(0.3)`
- `Linear(512 → C)`（`C` = 類別數 = 過濾後保留的罪名數）

多標籤損失：`BCEWithLogitsLoss`（logits 直接進 loss，不先做 sigmoid）。

---

## 標籤處理（Label Space）

- 只使用**訓練集**統計罪名頻率建立 `label2id`
- 過濾低頻罪名：`MIN_LABEL_FREQ=50`（可在 `config.py` 調整）
- 對 train/valid/test：
  - 若樣本有 `meta.accusation`，轉成 multi-hot 向量
  - 若樣本含標籤但全部都被低頻過濾掉，該樣本會被略過（避免全零標籤干擾訓練）
  - 若測試集無標籤（正式測試常見），則僅能做推論；本練習賽 test 內含標籤，因此可算 test 指標

輸出的 label map 會存到：

- `project/checkpoints/label_map.pt`

---

## 訓練設定與執行

### 主要超參數（`project/config.py`）

- `MAX_LEN = 512`
- `BATCH_SIZE = 4`
- `EPOCHS = 1`（目前設為 smoke test / 估時；要完整訓練請改回 10）
- differential LR：
  - `ENCODER_LR = 2e-5`
  - `HEAD_LR = 1e-3`
- `USE_GRADIENT_CHECKPOINTING = True`（預設開啟，省記憶體）
- `LOG_EVERY_N_STEPS = 50`（tqdm postfix 更新頻率）

### 裝置（CUDA / MPS / CPU）

程式會自動選擇裝置：CUDA > Apple MPS（M1/M2/M3/M4）> CPU。

需要強制指定時可在 `config.py` 設：

- `FORCE_DEVICE = "cpu"` 或 `"mps"` 或 `"cuda"`

### 執行方式（本機）

在 `project/` 目錄下執行：

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

> 第一次執行會下載 Hugging Face 權重與 tokenizer，需要連網；下載完成後通常可離線重跑（視快取是否存在）。

---

## 輸出檔案

訓練過程會以 **valid macro F1** 來挑選最佳 checkpoint：

- 最佳模型：`project/checkpoints/best_model.pt`
  - 內含 `model_state_dict`、`label2id/id2label`、`threshold`、`pretrained_model_name`、`tokenizer_name`、`num_classes`、`max_len`

推論：

```bash
cd project
python inference.py --text "輸入一段案情文字..."
```

---

## 評估指標

使用 `sklearn.metrics.f1_score` 計算：

- **macro F1**：對每個罪名算 F1 再平均（對低頻類敏感）
- **micro F1**：先加總 TP/FP/FN 再算（對高頻類更敏感）
- **subset accuracy**：每筆資料的 multi-hot 預測向量需**完全一致**才算對（多標籤下較嚴格）

---

## 實驗結果（1 Epoch, MacBook Air M4 / MPS）

環境：

- 裝置：`mps`（Apple GPU）
- Backbone：`hfl/chinese-legal-electra-small-discriminator`
- `MAX_LEN=512`, `BATCH_SIZE=4`, `EPOCHS=1`
- `MIN_LABEL_FREQ=50` → `num_classes=164`

資料規模（過濾後）：

- Train samples：153,745（38,437 batches/epoch）
- Valid samples：17,022（4,256 batches）
- Test samples：32,330（8,083 batches）

結果：

- **Valid**：macro F1 = **0.6809**，micro F1 = **0.8568**，subset accuracy = **0.7897**
- **Test**：macro F1 = **0.6678**，micro F1 = **0.8358**，subset accuracy = **0.7666**
- **總耗時**：**11693.8s（194.90 min）**
  - train 1 epoch：3:02:25（約 3.51 it/s）
  - valid：4:36
  - test：8:48

Checkpoint：

- `project/checkpoints/best_model.pt`

---

## 後續建議

- 將 `EPOCHS` 改回 10（或加入 early stopping），觀察 valid macro F1 是否繼續上升及是否過擬合。
- 若要縮短時間：可嘗試 `MAX_LEN=256` 或調整 `BATCH_SIZE`（在不 OOM 的前提下）。

