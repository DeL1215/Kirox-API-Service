from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# === 基本回應/請求 ===
class BaseResponse(BaseModel):
    message: str

class BaseUserIdRequest(BaseModel):
    userid: str

class BaseRobotIdRequest(BaseModel):
    robotid: str


# === 帳號相關 ===
class LoginRequest(BaseModel):
    gmail: EmailStr
    password: str

class LoginResponse(BaseResponse):
    userid: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    userid: str
    oldpassword: str
    newpassword: str

class RegisterRequest(BaseModel):
    gmail: EmailStr
    password: str
    nickname: str


# === data.py 需要的 models ===
class GetUserNicknameResponse(BaseModel):
    message: str
    nickname: Optional[str] = None


class RobotConfigResponse(BaseModel):
    robotid: str
    userid: str
    voiceid: Optional[int]
    voicename: Optional[str]
    promptstyle: Optional[str]
    robotname: Optional[str]
    status: Optional[bool]
    updatedtime: Optional[datetime]
    message: str


class VoiceItem(BaseModel):
    voiceid: int
    voicename: str
    description: Optional[str] = None  # 對應資料表可為 NULL

class VoiceListResponse(BaseModel):
    message: str
    voices: List[VoiceItem]


class RobotItem(BaseModel):
    robotid: str
    userid: str
    voiceid: Optional[int] = None
    voicename: Optional[str] = None
    promptstyle: Optional[str] = None
    robotname: Optional[str] = None
    status: Optional[bool] = None
    updatedtime: Optional[datetime] = None

class RobotListResponse(BaseModel):
    message: str
    robots: List[RobotItem]

# === 聊天相關 ===
class ChatMessage(BaseModel):
    chatid: Optional[int] = None
    robotid: str
    user_msg: Optional[str] = None
    tool_msg: Optional[str] = None
    ai_msg: Optional[str] = None
    image_base64: Optional[str] = None
    createdtime: Optional[str] = None  # 用字串方便外部（如 Milvus）處理

class AddChatRequest(BaseModel):
    robotid: str
    user_msg: Optional[str] = None
    tool_msg: Optional[str] = None
    ai_msg: Optional[str] = None
    image_base64: Optional[str] = None

class AddChatResponse(BaseModel):
    message: str
    chatid: Optional[int] = None

class GetChatHistoryRequest(BaseModel):
    robotid: str
    limit: Optional[int] = 20

class GetChatHistoryResponse(BaseModel):
    message: str
    history: List[ChatMessage]

class SearchChatRequest(BaseModel):
    robotid: str
    query_text: str
    limit: int = 5


# === 知識庫 ===
class AddKnowledgeRequest(BaseModel):
    robotid: str
    text: str
    title: Optional[str] = None
    source: Optional[str] = None

class AddKnowledgeResponse(BaseModel):
    message: str
    docid: Optional[int] = None

class KnowledgeChunk(BaseModel):
    docid: Optional[int] = None
    robotid: str
    text: str
    title: Optional[str] = None
    source: Optional[str] = None
    createdtime: Optional[str] = None  # ISO8601 字串

class SearchKnowledgeRequest(BaseModel):
    robotid: str
    query_text: str
    limit: int = 5

class SearchKnowledgeResponse(BaseModel):
    message: str
    items: List[KnowledgeChunk]

class GetKnowledgeListRequest(BaseModel):
    robotid: str
    limit: Optional[int] = 20

class GetKnowledgeListResponse(BaseModel):
    message: str
    items: List[KnowledgeChunk]


# === 單項更新請求（使用 Field 限制長度） ===
class UpdateVoiceRequest(BaseModel):
    robotid: str
    voiceid: int

class UpdateNameRequest(BaseModel):
    robotid: str
    robotname: str = Field(..., min_length=1, max_length=50)

class UpdatePromptStyleRequest(BaseModel):
    robotid: str
    promptstyle: str = Field(..., min_length=1, max_length=50)

# === 7 days chats count ===

class ChatStats7dRequest(BaseModel):
    robotid: str

class DailyCount(BaseModel):
    date: str   # 'YYYY-MM-DD' (台北當地日)
    count: int

class ChatStats7dResponse(BaseModel):
    message: str
    days: List[DailyCount]