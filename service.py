import logging
from pathlib import Path

from uvicorn.logging import AccessFormatter, DefaultFormatter

from fastapi import FastAPI
from routers import auth, data, chat, kb, camera

# Set up logging with timestamps and file output
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

CONSOLE_FORMAT = "[%(asctime)s] | %(levelprefix)s %(name)s %(message)s"
FILE_FORMAT = "[%(asctime)s] | %(levelname)s %(name)s %(message)s"

console_formatter = DefaultFormatter(CONSOLE_FORMAT, use_colors=True)
file_formatter = logging.Formatter(FILE_FORMAT)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(file_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler],
)

# 讓 uvicorn logger 帶時間戳並保留彩色輸出
uvicorn_error_formatter = DefaultFormatter(CONSOLE_FORMAT, use_colors=True)
uvicorn_access_formatter = AccessFormatter(
    '[%(asctime)s] | %(levelprefix)s %(client_addr)s "%(request_line)s" %(status_code)s'
)

for logger_name in ("uvicorn.error",):
    uv_logger = logging.getLogger(logger_name)
    uv_logger.setLevel(logging.INFO)
    uv_logger.propagate = True
    for handler in uv_logger.handlers:
        handler.setFormatter(uvicorn_error_formatter)

for logger_name in ("uvicorn.access",):
    uv_logger = logging.getLogger(logger_name)
    uv_logger.setLevel(logging.INFO)
    uv_logger.propagate = True
    for handler in uv_logger.handlers:
        handler.setFormatter(uvicorn_access_formatter)

logger = logging.getLogger(__name__)
logger.info("Logging initialized; writing to %s", LOG_FILE)

app = FastAPI(title="Kirox API", version="1.0")

API_PREFIX = "/api/v1"  # 想改成 /api 或 /v2 都行

app.include_router(auth.router, prefix=f"{API_PREFIX}/auth", tags=["auth"])
app.include_router(data.router, prefix=f"{API_PREFIX}/data", tags=["data"])
app.include_router(chat.router, prefix=f"{API_PREFIX}/chat", tags=["chat"])
app.include_router(kb.router,   prefix=f"{API_PREFIX}/kb",   tags=["kb"])
app.include_router(camera.router,   prefix=f"{API_PREFIX}/camera",   tags=["camera"])




#table:useraccount
#userid: uuid
#password: varchar
#updatetime: timestamp
#status: bool
#nickname: varchar
#gmail: varchar

#table:robot
#robotid: uuid
#robotnumber: varchar
#userid: uuid
#updatetime: timestamp
#status: bool
#styleprompt: varchar

#table:chathistory
#chatid: int4 自動遞增
#robotid: uuid
#createdtime: timestamp
#user_msg: varchar
#tool_msg: varchar
#ai_msg: varchar
#image_base64: varchar
