import logging
import time
from datetime import datetime, timezone
from threading import Thread

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    list_collections,
)
logger = logging.getLogger(__name__)

# === 基本連線設定（需要就自行調整） ===
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

# === Collection 名稱與向量維度 ===
CHAT_COLLECTION_NAME = "chat_memory"
KB_COLLECTION_NAME   = "kb_memory"

VECTOR_DIM = 512      # 兩個 collection 共用同維度（要改就一起改）

# === 對外暴露（保持舊相容性：collection = 聊天用） ===
collection = None       # chat collection（與你原本的 import 相容）
kb_collection = None    # knowledge base collection

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_collection(
    name: str,
    vector_dim: int,
    extra_fields: list,
    pk_name: str,
    need_dummy: bool = True
) -> Collection:
    """
    建立或載入指定名稱的 Collection，若空庫則插入一筆 dummy，並建立 IVF_FLAT + L2 索引與載入。
    extra_fields: 需包含 robotid / embedding / text / ... 等欄位（除了主鍵）。
    """
    # 1) 連線
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    # 2) 建立/載入
    if name in list_collections():
        col = Collection(name)
    else:
        fields = [FieldSchema(name=pk_name, dtype=DataType.INT64, is_primary=True, auto_id=False)]
        fields.extend(extra_fields)
        schema = CollectionSchema(fields, description=f"{name} collection")
        col = Collection(name, schema)
        logger.info("[Milvus] Collection [%s] 已建立，向量維度：%s", name, vector_dim)

    # 3) 空庫塞 dummy（好建索引）
    if col.num_entities == 0 and need_dummy:
        dummy_vec = [0.0] * vector_dim
        # 依不同 collection 的欄位順序插入
        if name == CHAT_COLLECTION_NAME:
            col.insert([
                [99999999],          # chatid
                ["dummy"],           # robotid
                [dummy_vec],         # embedding
                ["dummy for index"], # text
                [""],                # user_msg
                [""],                # tool_msg
                [""],                # ai_msg
                [""],                # image_base64
                [_now_iso()],        # createdtime
            ])
        else:
            # kb_memory 欄位順序（見下方 extra_fields 定義）
            col.insert([
                [99999999],          # docid
                ["dummy"],           # robotid
                [dummy_vec],         # embedding
                ["dummy for index"], # text（全文/摘要/切片）
                [""],                # title
                [""],                # source （檔名/URL/路徑）
                [_now_iso()],        # createdtime
            ])
        col.flush()
        logger.info("[Milvus] [%s] 已插入 Dummy 資料供建立索引", name)

    # 4) 建索引（如未建立）
    # 注意：如果你改 metric_type，search 時也要一致
    if not col.has_index():
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        col.create_index(field_name="embedding", index_params=index_params)
        logger.info("[Milvus] [%s] embedding index 建立完成！", name)

    # 5) 載入到記憶體
    col.load()

    # 6) 背景 flush
    def background_flush():
        while True:
            try:
                time.sleep(30)
                col.flush()
            except Exception as e:
                logger.exception("[Milvus Flush] %s 錯誤", name)

    Thread(target=background_flush, daemon=True).start()

    return col

def ensure_collections():
    global collection, kb_collection

    # --- 聊天用 collection（保持舊有相容性） ---
    chat_extra_fields = [
        FieldSchema(name="robotid",      dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="embedding",    dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="text",         dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="user_msg",     dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="tool_msg",     dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="ai_msg",       dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="createdtime",  dtype=DataType.VARCHAR, max_length=64),
    ]
    collection = _ensure_collection(
        name=CHAT_COLLECTION_NAME,
        vector_dim=VECTOR_DIM,
        extra_fields=chat_extra_fields,
        pk_name="chatid",
        need_dummy=True
    )

    # --- 知識庫用 collection（新增） ---
    # 簡潔設計：docid / robotid / embedding / text / title / source / createdtime
    kb_extra_fields = [
        FieldSchema(name="robotid",     dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="text",        dtype=DataType.VARCHAR, max_length=16384),  # 知識內容（可放 chunk）
        FieldSchema(name="title",       dtype=DataType.VARCHAR, max_length=512),    # 標題（可選）
        FieldSchema(name="source",      dtype=DataType.VARCHAR, max_length=1024),   # 來源（檔名/URL/路徑）
        FieldSchema(name="createdtime", dtype=DataType.VARCHAR, max_length=64),
    ]
    kb_collection = _ensure_collection(
        name=KB_COLLECTION_NAME,
        vector_dim=VECTOR_DIM,
        extra_fields=kb_extra_fields,
        pk_name="docid",
        need_dummy=True
    )

# 模組載入即確保兩個 collection 可用
ensure_collections()
