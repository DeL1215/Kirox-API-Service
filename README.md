# Kirox API Service
FastAPI 服務，涵蓋使用者/機器人管理、聊天紀錄語意檢索（Milvus）、知識庫維護，以及相機串流/截圖。

![alt text](background.png)
## 環境需求
- Python 3.10+（建議虛擬環境）
- PostgreSQL（預設連線在 `db.py`）
- Milvus 2.x（向量維度 512，L2 + IVF_FLAT）
- CUDA GPU（使用 `BAAI/bge-small-zh-v1.5` 產生 embedding；無 GPU 亦可但速度較慢）

## 快速開始
1) 建立環境並安裝依賴
```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn psycopg2-binary pymilvus sentence-transformers numpy pydantic
```

2) 準備後端服務
- PostgreSQL：確認資料庫 `Kirox-System` 以及相關資料表（`useraccount`, `robot`, `voice` 等）已就緒，連線設定在 `db.py`。
- Milvus：確保已啟動，且主機/連接埠符合 `milvus_helper.py`（預設 `127.0.0.1:19530`）。首次啟動時會自動建立 `chat_memory` 與 `kb_memory` collection 並建索引。

3) 啟動 API 伺服器
```bash
python -m uvicorn service:app --reload --host 0.0.0.0 --port 2025
```
瀏覽 `http://localhost:2025/docs` 查看互動式 API 文件。

## 主要模組與職責
- `service.py`：FastAPI 入口、路由註冊、日誌設定（輸出到 `logs/app.log`）。
- `routers/auth.py`：登入/註冊/變更密碼（PostgreSQL）。
- `routers/data.py`：機器人資訊與語音設定查改。
- `routers/chat.py`：聊天紀錄寫入、查詢、語意搜尋、7 天統計（Milvus）。
- `routers/kb.py`：知識庫新增/刪除/搜尋/列表（Milvus）。
- `routers/camera.py`：影像 WebSocket 上傳、MJPEG 串流、快照與在線列表。
- `embedding_model.py`：SentenceTransformer 背景執行緒批次產生 embedding。
- `milvus_helper.py`：確保 Milvus collection schema、索引與背景 flush。
- `security.py`：基本 SQL/XSS/路徑穿越檢查與格式驗證工具。

## API 概覽
- Auth：`POST /api/v1/auth/login`、`/register`、`/change-password`
- Robot 資料：`POST /api/v1/data/get-user-nickname`、`/get-robot-config`、`/get-robots-by-userid`、`PATCH /set-robot-voice|name|promptstyle`、`GET /get-voice-list`
- Chat：`POST /api/v1/chat/add-chat`、`/get-chat-history`、`/search-chat`、`/chat-stats-7d`、`DELETE /delete-chat/{chatid}`
- Knowledge Base：`POST /api/v1/kb/add-knowledge`、`/search-knowledge`、`/get-knowledge`、`DELETE /delete-knowledge/{docid}`
- Camera：`WS /api/v1/camera/upload/ws`、`GET /mjpeg`、`/snapshot`、`/robots/online`

## 設定重點
- PostgreSQL 連線：修改 `db.py` (`dbname/user/password/host/port`)。
- Milvus 連線與向量維度：修改 `milvus_helper.py` 中 `MILVUS_HOST`、`MILVUS_PORT`、`VECTOR_DIM`。
- 日誌位置：`logs/app.log`，可在 `service.py` 調整格式或路徑。

## 開發/除錯提示
- 需要持續寫入向量時，Milvus 背景 flush 每 30 秒一次；可手動呼叫 `collection.flush()` 以確保資料立即可查。
- embedding 維度需保持 512；若更換模型請同步更新 `VECTOR_DIM` 與斷言。
- Camera 相關端點以記憶體儲存最新影像，重啟服務後不保留歷史。

