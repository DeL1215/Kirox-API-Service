from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np
from datetime import datetime, timezone, timedelta
import zoneinfo


_TZ_TAIPEI = zoneinfo.ZoneInfo("Asia/Taipei")

from models import (
    AddChatRequest, AddChatResponse,
    GetChatHistoryRequest, GetChatHistoryResponse, ChatMessage, SearchChatRequest,ChatStats7dRequest,ChatStats7dResponse,DailyCount
)

# === Milvus 與 Embedding ===
from milvus_helper import collection  # 已在 import 時 ensure_collection()
from embedding_model import get_embedding

router = APIRouter()

# 回傳欄位（集中管理）
# 注意：這僅影響查詢輸出欄位，與 insert 的 schema 順序無關
OUTPUT_FIELDS = [
    "chatid", "robotid",
    "user_msg", "tool_msg", "ai_msg",
    "image_base64", "createdtime",
    "text",
]

# --------- 小工具：正規化與安全處理 ---------



def _parse_iso_to_dt_utc(ts: str) -> datetime:
    """
    把 createdtime (ISO8601, UTC) 字串轉成 datetime(aware, UTC)
    若壞字串就丟掉那筆
    """
    try:
        dt = datetime.fromisoformat(ts)
        # 保險：如果沒有 tzinfo，就當成 UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _dt_utc_to_taipei_date_str(dt_utc: datetime) -> str:
    """
    把 UTC datetime 轉成台北時間，回傳 'YYYY-MM-DD'
    """
    dt_local = dt_utc.astimezone(_TZ_TAIPEI)
    return dt_local.strftime("%Y-%m-%d")

def _iso_days_ago_utc(days: int) -> str:
    """
    取得現在往回 days 天的 UTC ISO8601 字串，用於 Milvus 篩 createdtime。
    例如 days=7 -> 現在-7天 的 UTC ISO 字串
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return cutoff.isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _gen_chatid() -> int:
    # 使用毫秒級時間戳產 PK（簡單、可讀、幾乎不撞）
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def _safe_str(v) -> str:
    # Milvus 的 string 欄位不接受 None
    return "" if v is None else str(v)

def _safe_robotid(robotid: str) -> str:
    # Milvus expr 中的字串要 escape 單引號
    return (robotid or "").replace("'", "''")

def _norm_vec(vec):
    # 確保 float32 list，避免 dtype 或 numpy array 直接丟入導致不相容
    return np.asarray(vec, dtype=np.float32).tolist()

# --------- 轉換工具 ---------
def _hit_to_chatmessage(hit) -> ChatMessage:
    ent = hit.entity
    return ChatMessage(
        chatid=int(ent.get("chatid")),
        robotid=ent.get("robotid"),
        user_msg=ent.get("user_msg"),
        tool_msg=ent.get("tool_msg"),
        ai_msg=ent.get("ai_msg"),
        image_base64=ent.get("image_base64"),
        createdtime=ent.get("createdtime"),
    )

def _row_to_chatmessage(row: dict) -> ChatMessage:
    return ChatMessage(
        chatid=int(row.get("chatid")),
        robotid=row.get("robotid"),
        user_msg=row.get("user_msg"),
        tool_msg=row.get("tool_msg"),
        ai_msg=row.get("ai_msg"),
        image_base64=row.get("image_base64"),
        createdtime=row.get("createdtime"),
    )

# --------- 路由 ---------
@router.post("/search-chat", response_model=GetChatHistoryResponse)
def search_chat(data: SearchChatRequest):
    """
    以語義搜尋相似對話（僅用 Milvus）：
      - 先把 query_text 轉 embedding
      - 以 robotid 過濾，再依相似度排序回傳
    """
    query_vec = _norm_vec(get_embedding(data.query_text))
    safe_robot = _safe_robotid(data.robotid)

    try:
        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},  # 與索引設定一致
            limit=data.limit or 5,
            expr=f"robotid == '{safe_robot}'",
            output_fields=OUTPUT_FIELDS
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus search failed: {e}")

    if not results or len(results[0]) == 0:
        return GetChatHistoryResponse(message="沒有找到相似對話", history=[])

    history: List[ChatMessage] = [_hit_to_chatmessage(hit) for hit in results[0]]
    return GetChatHistoryResponse(message="語意查詢成功", history=history)

@router.post("/get-chat-history", response_model=GetChatHistoryResponse)
def get_chat_history(data: GetChatHistoryRequest):
    """
    取某 robotid 歷史對話：
      - 從 Milvus 把該 robotid 的資料全部拉回來（上限 16384）
      - 依 createdtime 由新到舊排序
      - 回傳「新 -> 舊」
    """
    limit = getattr(data, "limit", None) or 20
    safe_robot = _safe_robotid(data.robotid)

    try:
        raw = collection.query(
            expr=f"robotid == '{safe_robot}'",
            output_fields=OUTPUT_FIELDS,
            # Milvus query offset+limit 上限 16384，這裡直接全吃
            limit=16384,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus query failed: {e}")

    if not raw:
        return GetChatHistoryResponse(message="沒有資料", history=[])

    # createdtime 由新到舊
    raw.sort(key=lambda r: r.get("createdtime") or "", reverse=True)

    # 只拿前 limit 筆
    picked = raw[:limit]

    # 這裡不要再 reverse() 了，維持「新 -> 舊」
    history = [_row_to_chatmessage(r) for r in picked]
    return GetChatHistoryResponse(message="取得聊天紀錄成功", history=history)


@router.post("/add-chat", response_model=AddChatResponse)
def add_chat(data: AddChatRequest):
    """
    寫入一輪對話（Milvus only）：
      - chatid: 毫秒級時間戳
      - createdtime: ISO8601 (UTC)
      - text: user/tool/ai 三段合併，供語義檢索
    """
    if not data.robotid:
        raise HTTPException(status_code=400, detail="robotid 不可為空")

    chatid = _gen_chatid()
    created = _now_iso()

    # 組語義文本（全部轉字串且容忍 None）
    text = " ".join([
        _safe_str(data.user_msg),
        _safe_str(data.tool_msg),
        _safe_str(data.ai_msg),
    ]).strip()

    embedding = _norm_vec(get_embedding(text))

    # --- 關鍵修正：所有 string 欄位都保證不是 None ---
    user_msg = _safe_str(data.user_msg)
    tool_msg = _safe_str(data.tool_msg)          # ★ 避免 None → ParamError
    ai_msg   = _safe_str(data.ai_msg)
    img_b64  = _safe_str(data.image_base64)

    try:
        # 插入順序需與 collection schema 完全一致（請確認你的建表順序）
        # 假設 schema 順序：chatid, robotid, embedding, text, user_msg, tool_msg, ai_msg, image_base64, createdtime
        collection.insert([
            [chatid],           # chatid
            [data.robotid],     # robotid
            [embedding],        # embedding
            [text],             # text
            [user_msg],         # user_msg
            [tool_msg],         # tool_msg
            [ai_msg],           # ai_msg
            [img_b64],          # image_base64
            [created],          # createdtime
        ])
        # 視需求：collection.flush()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus insert failed: {e}")

    return AddChatResponse(message="一輪對話已新增", chatid=chatid)

@router.delete("/delete-chat/{chatid}", response_model=AddChatResponse)
def delete_chat(chatid: int):
    """
    刪除指定 chatid（Milvus）
    """
    expr = f"chatid == {chatid}"
    try:
        mr = collection.delete(expr=expr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus delete failed: {e}")

    # 嘗試確認是否刪成功（不同版本回傳略有差異）
    try:
        if hasattr(mr, "delete_count") and mr.delete_count == 0:
            raise HTTPException(status_code=404, detail="找不到該 chatid")
    except Exception:
        try:
            remains = collection.query(expr=expr, output_fields=["chatid"], limit=1)
            if remains:
                raise HTTPException(status_code=500, detail="刪除未生效，請稍後再試")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Milvus query failed after delete: {e}")

    return AddChatResponse(message=f"已刪除 chatid={chatid}")

@router.post("/chat-stats-7d", response_model=ChatStats7dResponse)
def chat_stats_7d(data: ChatStats7dRequest):
    """
    回傳這個 robot 在最近 7 天內的每日對話次數 (依台北當地日期分組)。

    作法：
    1. 算出 UTC 現在往回 7 天的 ISO 時間字串 cutoff_iso
    2. 用 Milvus query:
         - robotid == 'xxx'
         - createdtime >= cutoff_iso   (因為是 ISO8601 UTC，字典序 ≈ 時間序)
    3. 取回 rows 後：
         - 把每筆 createdtime parse 成 UTC datetime
         - 轉台北時間，取當地日期字串 'YYYY-MM-DD'
         - 做 counter
    4. 確保完全覆蓋「今天往前 6 天」，即共 7 天
         - 沒資料的日子 count = 0 也要回
    """

    safe_robot = _safe_robotid(data.robotid)

    # 7 天前 (含今天共 7 天): 我們先抓「現在-7天」當 cutoff
    # 例如今天 2025-10-29，cutoff 就是 2025-10-22T...Z
    cutoff_iso = _iso_days_ago_utc(7)

    expr = (
        f"robotid == '{safe_robot}' && createdtime >= '{cutoff_iso}'"
    )

    try:
        # 把近 7 天內的資料全部拉回來，量通常不會爆到幾十萬
        raw = collection.query(
            expr=expr,
            output_fields=["createdtime"],
            limit=10000  # 防呆上限，可調
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Milvus query failed: {e}")

    # --- 統計 ---
    # 先做一個 dict: { 'YYYY-MM-DD': 次數 }
    counter = {}

    for r in raw:
        ts = r.get("createdtime")
        dt_utc = _parse_iso_to_dt_utc(ts)
        if dt_utc is None:
            continue
        d_local = _dt_utc_to_taipei_date_str(dt_utc)
        counter[d_local] = counter.get(d_local, 0) + 1

    # --- 我們要保證連續 7 天的日期都有，就算 0 ---
    # 定義「今天」用台北時間的今天日期
    today_local = datetime.now(timezone.utc).astimezone(_TZ_TAIPEI).date()

    day_list = []
    for i in range(0, 7):
        day_date = today_local - timedelta(days=i)
        day_str = day_date.strftime("%Y-%m-%d")
        c = counter.get(day_str, 0)
        day_list.append({"date": day_str, "count": c})

    # 現在 day_list[0] 是今天，day_list[6] 是 6 天前
    # 你如果想「舊 -> 新」，就 reverse
    day_list.reverse()

    # 組 Response
    return ChatStats7dResponse(
        message="近七天對話次數統計完成",
        days=[DailyCount(**d) for d in day_list]
    )
