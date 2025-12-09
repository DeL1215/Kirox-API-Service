from fastapi import APIRouter, HTTPException
from psycopg2.extras import RealDictCursor
from db import get_connection
from models import (
    BaseRobotIdRequest,
    GetUserNicknameResponse,
    RobotConfigResponse,
    VoiceListResponse,
    UpdateVoiceRequest,
    UpdateNameRequest,
    UpdatePromptStyleRequest,
    RobotListResponse,
    BaseUserIdRequest
)
from security import is_input_safe_for_sql, validate_uuid

router = APIRouter()


# ------------------------
# 共用檢查/查詢函式
# ------------------------
def _check_robotid(robotid: str):
    """檢查 robotid 格式與安全性"""
    if not validate_uuid(robotid):
        raise HTTPException(status_code=400, detail="robotid 格式不合法")
    safe, reasons = is_input_safe_for_sql(robotid)
    if not safe:
        raise HTTPException(status_code=400, detail=f"robotid 包含非法輸入: {reasons}")


def _fetch_robot_config(cursor, robotid: str) -> dict | None:
    """讀取單一機器人的完整配置（含 voice join）"""
    cursor.execute(
        """
        SELECT
            r.robotid,
            r.userid,
            r.voiceid,
            v.voicename,
            r.promptstyle,
            r.robotname,
            r.status,
            r.updatedtime
        FROM robot r
        LEFT JOIN voice v ON v.voiceid = r.voiceid
        WHERE r.robotid = %s
        """,
        (robotid,),
    )
    return cursor.fetchone()


def _ensure_voice_exists(cursor, voiceid: int):
    """若 DB 未設外鍵，建議在應用層先驗證 voice 是否存在"""
    cursor.execute("SELECT 1 FROM voice WHERE voiceid = %s", (voiceid,))
    if cursor.fetchone() is None:
        raise HTTPException(status_code=400, detail="voiceid 不存在")


# ------------------------
# 查詢類 API
# ------------------------
@router.post("/get-user-nickname", response_model=GetUserNicknameResponse)
def get_user_nickname(data: BaseUserIdRequest):
    if not validate_uuid(data.userid):
        raise HTTPException(status_code=400, detail="userid 格式不合法")
    safe, reasons = is_input_safe_for_sql(data.userid)
    if not safe:
        raise HTTPException(status_code=400, detail=f"userid 包含非法輸入: {reasons}")
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        """
        SELECT ua.nickname
        FROM useraccount ua
        WHERE ua.userid = %s
        """,
        (data.userid,),
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="找不到對應的使用者")

    return GetUserNicknameResponse(message="查詢成功", nickname=result["nickname"])


@router.post("/get-robot-config", response_model=RobotConfigResponse)
def get_robot_config(data: BaseRobotIdRequest):
    _check_robotid(data.robotid)

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        """
        SELECT
            r.robotid,
            r.userid,
            r.voiceid,
            v.voicename,
            r.promptstyle,
            r.robotname,
            r.status,
            r.updatedtime
        FROM robot r
        LEFT JOIN voice v ON v.voiceid = r.voiceid
        WHERE r.robotid = %s
        """,
        (data.robotid,),
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="找不到對應的機器人配置")

    result["message"] = "查詢成功"
    return result


@router.post("/get-robots-by-userid", response_model=RobotListResponse)
def get_robots_by_userid(data: BaseUserIdRequest):
    if not validate_uuid(data.userid):
        raise HTTPException(status_code=400, detail="userid 格式不合法")
    safe, reasons = is_input_safe_for_sql(data.userid)
    if not safe:
        raise HTTPException(status_code=400, detail=f"userid 包含非法輸入: {reasons}")

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                r.robotid,
                r.userid,
                r.voiceid,
                v.voicename,
                r.promptstyle,
                r.robotname,
                r.status,
                r.updatedtime
            FROM robot r
            LEFT JOIN voice v ON v.voiceid = r.voiceid
            WHERE r.userid = %s
            ORDER BY r.updatedtime DESC
            """,
            (data.userid,),
        )
        rows = cursor.fetchall() or []
        return {"message": "查詢成功", "robots": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢失敗: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.get("/get-voice-list", response_model=VoiceListResponse)
def get_voice_list():
    """
    取得 voice 表的完整清單。
    回傳欄位：voiceid, voicename, description
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT voiceid, voicename, description
            FROM voice
            ORDER BY voiceid
            """
        )
        rows = cursor.fetchall() or []
        return VoiceListResponse(message="查詢成功", voices=rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢失敗: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ------------------------
# 單欄位更新 API（分開）
# ------------------------
@router.patch("/set-robot-voice", response_model=RobotConfigResponse)
def set_robot_voice(data: UpdateVoiceRequest):
    _check_robotid(data.robotid)

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        _ensure_voice_exists(cursor, data.voiceid)

        cursor.execute(
            """
            UPDATE robot
            SET voiceid = %s, updatedtime = NOW()
            WHERE robotid = %s
            """,
            (data.voiceid, data.robotid),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="找不到對應的機器人")

        conn.commit()

        result = _fetch_robot_config(cursor, data.robotid)
        if not result:
            raise HTTPException(status_code=404, detail="找不到對應的機器人配置")
        result["message"] = "更新成功"
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失敗: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.patch("/set-robot-name", response_model=RobotConfigResponse)
def set_robot_name(data: UpdateNameRequest):
    _check_robotid(data.robotid)

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute(
            """
            UPDATE robot
            SET robotname = %s, updatedtime = NOW()
            WHERE robotid = %s
            """,
            (data.robotname, data.robotid),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="找不到對應的機器人")

        conn.commit()

        result = _fetch_robot_config(cursor, data.robotid)
        if not result:
            raise HTTPException(status_code=404, detail="找不到對應的機器人配置")
        result["message"] = "更新成功"
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失敗: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.patch("/set-robot-promptstyle", response_model=RobotConfigResponse)
def set_robot_promptstyle(data: UpdatePromptStyleRequest):
    _check_robotid(data.robotid)

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute(
            """
            UPDATE robot
            SET promptstyle = %s, updatedtime = NOW()
            WHERE robotid = %s
            """,
            (data.promptstyle, data.robotid),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="找不到對應的機器人")

        conn.commit()

        result = _fetch_robot_config(cursor, data.robotid)
        if not result:
            raise HTTPException(status_code=404, detail="找不到對應的機器人配置")
        result["message"] = "更新成功"
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失敗: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
