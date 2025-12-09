import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

# 建立 Router
router = APIRouter()
logger = logging.getLogger(__name__)

# 用來存每台機器人最新的一張影像
# key: (robot_id, camera_id)
# value: {"frame": bytes, "timestamp": float}
frames: Dict[Tuple[str, str], Dict[str, Any]] = {}

# MJPEG 的 boundary 名稱
MJPEG_BOUNDARY = "frame"


@router.websocket("/upload/ws")
async def camera_upload_ws(
    websocket: WebSocket,
    robot_id: str = Query(..., description="機器人唯一 ID，例如 kirox-001"),
    camera_id: str = Query("default", description="相機 ID，例如 front / top，預設 default"),
):
    """
    Jetson / ROS2 Camera Node 透過這個 WebSocket 上傳 JPEG 圖片。

    - 連線網址範例：
      wss://api.xbotworks.com/api/v1/camera/upload/ws?robot_id=kirox-001&camera_id=front
    - 接收的資料格式：純 JPEG bytes（一張一張送）
    """
    await websocket.accept()
    key = (robot_id, camera_id)
    logger.info("[WS UPLOAD] connected: robot_id=%s, camera_id=%s", robot_id, camera_id)

    try:
        while True:
            # 等待 client 傳來一塊 binary（JPEG）
            data = await websocket.receive_bytes()

            # 更新最新影像與時間戳
            frames[key] = {
                "frame": data,
                "timestamp": time.time(),
            }

    except WebSocketDisconnect as e:
        # 把 close code 印出來，方便你之後判斷是正常關還是被代理砍
        code = getattr(e, "code", None)
        logger.info(
            "[WS UPLOAD] disconnected: robot_id=%s, camera_id=%s, code=%s",
            robot_id,
            camera_id,
            code,
        )
    except Exception as e:
        logger.exception(
            "[WS UPLOAD] error for robot_id=%s, camera_id=%s", robot_id, camera_id
        )


async def mjpeg_generator(robot_id: str, camera_id: str = "default"):
    """
    低延遲、減少閃爍的 MJPEG 串流：

    設計：
    - 只在「有新 frame」時才送，避免一直重送舊圖造成 App 閃爍
    - 控制最大串流 FPS（max_stream_fps）
    - 永遠只拿 frames 裡「當下最新」的一張，不排隊、不堆 buffer
    - 若長時間沒有任何新 frame，可設定 timeout 中止串流
    """
    key = (robot_id, camera_id)

    # 最長多久沒有任何新 frame 就中止串流（秒；0 表示不超時）
    stream_timeout_seconds = 60

    # 串流輸出的 FPS（給瀏覽器/App 的頻率，不是 Jetson 真實 FPS）
    max_stream_fps = 10.0
    if max_stream_fps <= 0:
        max_stream_fps = 1.0
    min_interval = 1.0 / max_stream_fps

    last_send_time: float = 0.0
    last_frame_ts: float = 0.0
    last_valid_time: float = time.time()

    while True:
        now = time.time()

        # timeout：太久沒有任何新 frame → 中止串流，交給 client 決定是否重連
        if stream_timeout_seconds > 0 and now - last_valid_time > stream_timeout_seconds:
            logger.warning("[MJPEG] timeout: robot_id=%s, camera_id=%s", robot_id, camera_id)
            break

        item = frames.get(key)
        if item is None:
            # Jetson 還沒開始上傳或暫停中
            await asyncio.sleep(0.1)
            continue

        frame: bytes = item["frame"]
        ts: float = item.get("timestamp", 0.0)

        # 若沒有「新」 frame，就稍微睡一下再看（不送任何東西，避免閃爍）
        if ts == last_frame_ts:
            await asyncio.sleep(0.01)
            continue

        # 有新 frame 了，更新「最後有效時間」與 last_frame_ts
        last_frame_ts = ts
        last_valid_time = now

        # 控制對外串流 FPS：距離上次送出時間太短就等一下
        elapsed = now - last_send_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
            # 等完後再 loop 一次，此時 frames 可能又更新了，只會送最新的
            continue

        last_send_time = time.time()

        # 建立 multipart header
        header = (
            f"--{MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n"
            "\r\n"
        ).encode("utf-8")

        # 送出「這一幀最新」的影像
        yield header + frame + b"\r\n"


@router.get("/mjpeg")
async def camera_mjpeg(
    robot_id: str = Query(..., description="要觀看的機器人 ID，例如 kirox-001"),
    camera_id: str = Query("default", description="要觀看的相機 ID，預設 default"),
):
    """
    提供 MJPEG 串流，給 WebView 或瀏覽器直接觀看。

    - 不先檢查 frames 是否已有資料，避免一開始就 404
    - client 連上後會掛著等，只要 Jetson 開始上傳就會有畫面
    """
    return StreamingResponse(
        mjpeg_generator(robot_id, camera_id),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )


@router.get("/snapshot")
async def camera_snapshot(
    robot_id: str = Query(..., description="要看的機器人 ID，例如 kirox-001"),
    camera_id: str = Query("default", description="相機 ID，預設 default"),
):
    """
    取得目前最新的一張 JPG（給不支援 MJPEG 的 client 用）。

    - App 若是 Flutter / React Native / 原生 Android，
      可以用這個 endpoint 輪詢（例如每秒 5~10 次），
      通常會比自己處理 MJPEG 還穩，不容易閃爍。
    """
    key = (robot_id, camera_id)
    item = frames.get(key)

    if item is None:
        raise HTTPException(status_code=404, detail="No frame available for this robot/camera yet")

    frame: bytes = item["frame"]
    return Response(content=frame, media_type="image/jpeg")


@router.get("/robots/online")
async def robots_online(threshold_seconds: int = 10):
    """
    回傳目前「有在送畫面」的機器人列表。

    - threshold_seconds：多久內有更新就算 online（預設 10 秒）
    - 回傳格式：
      [
        {"robot_id": "kirox-001", "camera_id": "front", "last_update": 1234567890.0},
        ...
      ]
    """
    now = time.time()
    online: List[Dict[str, Any]] = []

    for (robot_id, camera_id), info in frames.items():
        ts = info.get("timestamp", 0)
        if now - ts <= threshold_seconds:
            online.append(
                {
                    "robot_id": robot_id,
                    "camera_id": camera_id,
                    "last_update": ts,
                }
            )

    return JSONResponse(content=online)
