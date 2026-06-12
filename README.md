# 本地端小說翻譯協作系統

使用 Python、Gradio、Ollama 與 Llama-3-Taiwan-8B-Instruct 製作的英翻繁中
原型。系統以人類可讀寫的 `glossary.md` 保存譯名、稱謂與角色語氣。

模型只提出術語候選；候選會在網頁中顯示成表格，必須由使用者批准後，程式
才會更新 Markdown。第一版的目標是讓完整工作流程可執行，不進行模型比較。

## 功能

- 上傳或貼上英文小說章節
- 掃描人物、地名、組織、物件、能力與稱謂
- 使用 spaCy 找出英文實體，再由 Llama 分類並產生繁中譯名
- 以網頁表格審核術語候選
- 依 `glossary.md` 分段翻譯
- 審查漏譯、誤譯、譯名違規、代詞錯置與翻譯腔
- 匯出繁中譯文與更新後的 `glossary.md`
- 從 Project Gutenberg 純文字網址下載並清理公版測試文本

## 系統需求

- Python 3.10 以上
- 16GB RAM 建議值
- [Ollama](https://ollama.com/)
- Llama-3-Taiwan-8B-Instruct 的 GGUF 量化檔

CPU 可以執行，但速度依處理器、量化版本和文本長度而異。

## 安裝 Python 環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 安裝模型

1. 前往 [Llama-3-Taiwan-8B-Instruct 官方模型頁](https://huggingface.co/yentinglin/Llama-3-Taiwan-8B-Instruct)。
2. 點選 **Browse Quantizations**，選擇可信來源的 `Q4_K_M` GGUF。
3. 將檔名改為 `Llama-3-Taiwan-8B-Instruct-Q4_K_M.gguf`。
4. 放到本專案的 `models/` 目錄。
5. 在專案根目錄執行：

```powershell
ollama create llama-3-taiwan-8b -f Modelfile
```

確認模型可用：

```powershell
ollama run llama-3-taiwan-8b "請用繁體中文回覆：測試成功"
```

GGUF 通常由第三方量化，下載前應核對來源、模型基底與授權。模型檔很大，
不應提交至 Git。

## 啟動

確認 Ollama 桌面程式或 `ollama serve` 正在執行，再啟動網頁：

```powershell
python app.py
```

瀏覽器會開啟本機 Gradio 網址。程式預設只連線到
`http://localhost:11434`，不會主動把小說內容送到雲端。

## 使用流程

1. 貼上英文原文，或上傳 UTF-8 `.txt`／`.md`。
2. 視需要載入上一章匯出的 `glossary.md`。
3. 按「掃描術語候選」。
4. 在候選表格中修正內容，並刪除不採用的列。
5. 按「批准表格中的術語」，更新畫面中的 `glossary.md`。
6. 按「產生翻譯初稿」。
7. 檢閱初稿後，按「執行品質審查」。
8. 人工確認審查稿，再匯出譯文與術語表。

`examples/` 內有可立即使用的英文短篇與範例術語表。

## 下載公版測試文本

網頁的「下載 Project Gutenberg 測試文本」可直接貼入 Plain Text UTF-8
網址。也可使用命令列：

```powershell
python download_gutenberg.py `
  "https://www.gutenberg.org/cache/epub/55/pg55.txt" `
  -o "test-data/wizard-of-oz.txt"
```

工具會移除 Project Gutenberg 頁首與授權頁尾，但不會改寫小說正文。若需保存
完整原始檔，可加上 `--keep-boilerplate`。

## 測試

不啟動 Ollama 也能執行程式邏輯測試：

```powershell
python -m pytest
```

這些測試使用固定回應的測試替身，只檢查 Markdown 解析、術語批准、長文分段
與流程串接。它不是翻譯模型，也不代表真實翻譯品質。

完成模型安裝後，請依「確認模型可用」命令及 `examples/` 再執行一次真實模型
smoke test。

## NER 實驗

第一輪只需準備一份 UTF-8 英文文本，直接比較不同方法抽出的字詞：

```powershell
python benchmark_ner.py test-data/wizard-of-oz.txt `
  --methods spacy spacy-propn `
  -o test-data/ner-comparison.json
```

報告會列出各方法的完整候選、共同字詞，以及各自獨有的字詞。

若之後要計算 precision、recall 與 F1，再準備人工標註 JSON：

```json
[
  {"text": "Dorothy", "type": "人物"},
  {"text": "Kansas", "type": "地名"}
]
```

可用 `examples/ner-gold-template.json` 作為起點。執行 spaCy 比較：

```powershell
python benchmark_ner.py test-data/wizard-of-oz.txt `
  --gold test-data/wizard-of-oz-gold.json `
  --methods spacy spacy-propn `
  -o test-data/ner-report.json
```

安裝 GLiNER 後可加入第三種方法：

```powershell
python -m pip install gliner
python benchmark_ner.py test-data/wizard-of-oz.txt `
  --methods spacy spacy-propn gliner
```

GLiNER 會自動將長文切成 220 個非空白 token 的片段，並讓相鄰片段重疊
40 個 token，避免超過模型長度限制。可用 `--gliner-chunk-size` 與
`--gliner-chunk-overlap` 調整。

主指標只比較實體名稱是否被抓到；類型判斷錯誤會另外列出。完整 JSON 報告包含
每種方法的候選、漏抓、誤抓、類型錯誤與執行時間。

## 限制與倫理

- 第一版只支援英文小說翻譯成臺灣繁體中文。
- 不處理漫畫 OCR、圖像脈絡、排版或日文翻譯。
- LLM 可能誤譯、漏譯或提出錯誤術語，發布前必須人工校閱。
- 分段翻譯可能在段落邊界失去局部脈絡。
- 只應處理你擁有、已獲授權或依法可使用的文本。
- 翻譯權與散布權可能各自獨立；本工具不鼓勵未授權發布。
- 本專案不宣稱 AI 輸出等同專業譯者成品。

## 專案定位

本原型探討低成本本地模型能否透過人類可編輯的外部術語記憶，降低長篇翻譯
初稿與一致性管理的負擔。重點是文化近用、文本隱私、低硬體門檻與人機協作，
不是以「小模型戰勝大型模型」作為預設結論。

## 授權

程式碼採 MIT License。範例文本為本專案自製。使用者輸入文本的權利仍歸原
權利人；Llama-3-Taiwan 權重另受其模型授權約束。
