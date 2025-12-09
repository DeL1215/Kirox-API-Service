from fastapi import APIRouter, HTTPException
from psycopg2.extras import RealDictCursor
from db import get_connection
from models import LoginRequest, LoginResponse, ChangePasswordRequest, RegisterRequest, BaseResponse
from uuid import uuid4
from datetime import date

router = APIRouter()

@router.post("/login", response_model=LoginResponse)
def login(data: LoginRequest):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT password, status, userid FROM useraccount WHERE gmail = %s", (data.gmail,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        raise HTTPException(status_code=404, detail="帳號不存在")
    if user["password"] != data.password:
        raise HTTPException(status_code=401, detail="密碼錯誤")
    if not user["status"]:
        raise HTTPException(status_code=403, detail="帳號尚未啟用")

    return LoginResponse(message="登入成功", userid=user["userid"])

@router.post("/change-password", response_model=BaseResponse)
def change_password(data: ChangePasswordRequest):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT password FROM useraccount WHERE userid = %s", (data.userid,))
    user = cursor.fetchone()
    if not user:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="找不到使用者")
    if user["password"] != data.oldpassword:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=401, detail="舊密碼錯誤")

    cursor.execute("UPDATE useraccount SET password = %s, updatetime = %s WHERE userid = %s",
                   (data.newpassword, date.today(), data.userid))
    conn.commit()
    cursor.close()
    conn.close()
    return BaseResponse(message="密碼已成功變更")

@router.post("/register", response_model=BaseResponse)
def register(data: RegisterRequest):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT gmail FROM useraccount WHERE gmail = %s", (data.gmail,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        raise HTTPException(status_code=409, detail="信箱已被註冊")

    new_userid = str(uuid4())
    cursor.execute("""
        INSERT INTO useraccount (userid, gmail, password, nickname, status, updatetime)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (new_userid, data.gmail, data.password, data.nickname, True, date.today()))
    conn.commit()
    cursor.close()
    conn.close()
    return BaseResponse(message="註冊成功")