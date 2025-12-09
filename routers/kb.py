from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timezone
from typing import List
import numpy as np

from models import (
    AddKnowledgeRequest, AddKnowledgeResponse, BaseResponse,
    SearchKnowledgeRequest, SearchKnowledgeResponse, KnowledgeChunk,
    GetKnowledgeListRequest, GetKnowledgeListResponse
)
from milvus_helper import kb_collection
from embedding_model import get_embedding

router = APIRouter()

# 需要回傳的欄位（與 kb schema 對齊）
KB_OUTPUT_FIELDS = ["docid", "robotid", "text", "title", "source", "createdtime"]

# --------- 小工具 ---------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _gen_docid() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def _safe_str(v) -> str:
    return "" if v is None else str(v)

def _safe_robotid(robotid: str) -> str:
    return (robotid or "").replace("'", "''")

def _norm_vec(vec):
    return np.asarray(vec, dtype=np.float32).tolist()

# --------- 轉換工具 ---------
def _hit_to_chunk(hit) -> KnowledgeChunk:
    ent = hit.entity
    return KnowledgeChunk(
        docid=int(ent.get("docid")),
        robotid=ent.get("robotid"),
        text=ent.get("text"),
        title=ent.get("title"),
        source=ent.get("source"),
        createdtime=ent.get("createdtime"),
    )

def _row_to_chunk(row: dict) -> KnowledgeChunk:
    return KnowledgeChunk(
        docid=int(row.get("docid")),
        robotid=row.get("robotid"),
        text=row.get("text"),
        title=row.get("title"),
        source=row.get("source"),
        createdtime=row.get("createdtime"),
    )

# --------- 路由 ---------
@router.post("/add-knowledge", response_model=AddKnowledgeResponse)
def add_knowledge(data: AddKnowledgeRequest):
    if not data.robotid:
        raise HTTPException(status_code=400, detail="robotid 不可為空")
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="text 不可為空")

    docid = _gen_docid()
    created = _now_iso()
    embedding = _norm_vec(get_embedding(data.text))

    title = _safe_str(data.title)
    source = _safe_str(data.source)

    try:
        # 插入順序需與 kb_collection schema 完全一致：
        # docid, robotid, embedding, text, title, source, createdtime
        kb_collection.insert([
            [docid],
            [data.robotid],
            [embedding],
            [data.text],
            [title],
            [source],
            [created],
        ])
        # 視需求：kb_collection.flush()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus insert failed: {e}")

    return AddKnowledgeResponse(message="知識已新增", docid=docid)

@router.delete("/delete-knowledge/{docid}", response_model=BaseResponse)
def delete_knowledge(docid: int, robotid: str = Query(..., description="為避免刪到他人的資料，必須帶上 robotid")):
    safe_robot = _safe_robotid(robotid)

    try:
        exists = kb_collection.query(
            expr=f"docid == {docid} && robotid == '{safe_robot}'",
            output_fields=["docid"],
            limit=1
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus query failed: {e}")

    if not exists:
        raise HTTPException(status_code=404, detail="找不到此 docid 或不屬於該 robotid")

    try:
        kb_collection.delete(expr=f"docid == {docid} && robotid == '{safe_robot}'")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus delete failed: {e}")

    try:
        remains = kb_collection.query(
            expr=f"docid == {docid} && robotid == '{safe_robot}'",
            output_fields=["docid"], limit=1
        )
        if remains:
            raise HTTPException(status_code=500, detail="刪除未生效，請稍後再試")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus query failed after delete: {e}")

    return BaseResponse(message=f"已刪除 docid={docid}")

@router.post("/search-knowledge", response_model=SearchKnowledgeResponse)
def search_knowledge(data: SearchKnowledgeRequest):
    """
    語意搜尋（與對話搜尋一致的邏輯）：
    - 用 query_text 取 embedding
    - 以 robotid 過濾
    - 依相似度排序回傳
    """
    query_vec = _norm_vec(get_embedding(data.query_text))
    safe_robot = _safe_robotid(data.robotid)

    try:
        results = kb_collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=data.limit or 5,
            expr=f"robotid == '{safe_robot}'",
            output_fields=KB_OUTPUT_FIELDS
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus search failed: {e}")

    if not results or len(results[0]) == 0:
        return SearchKnowledgeResponse(message="沒有找到相似知識", items=[])

    items: List[KnowledgeChunk] = [_hit_to_chunk(hit) for hit in results[0]]
    return SearchKnowledgeResponse(message="語意查詢成功", items=items)

@router.post("/get-knowledge", response_model=GetKnowledgeListResponse)
def get_knowledge(data: GetKnowledgeListRequest):
    """
    拉取某 robotid 的知識清單（最新 N 筆），新 -> 舊
    """
    limit = getattr(data, "limit", None) or 20
    safe_robot = _safe_robotid(data.robotid)

    try:
        raw = kb_collection.query(
            expr=f"robotid == '{safe_robot}'",
            output_fields=KB_OUTPUT_FIELDS,
            limit=16384,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus query failed: {e}")

    if not raw:
        return GetKnowledgeListResponse(message="沒有資料", items=[])

    # 新 -> 舊
    raw.sort(key=lambda r: r.get("createdtime") or "", reverse=True)
    picked = raw[:limit]

    items = [_row_to_chunk(r) for r in picked]
    return GetKnowledgeListResponse(message="取得知識清單成功", items=items)
