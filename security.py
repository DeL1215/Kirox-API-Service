# security.py
# 安全輸入檢查工具集
# 提供：SQL 注入 / XSS / 路徑穿越 / 控制字元檢查，與若干型別驗證函式
# 注意：此為快速 guard，**不可**取代使用參數化 SQL (prepared statements)。

import re
import html
import json
from typing import Optional, Tuple, List
from uuid import UUID

# ---------- 模式定義（移除 inline (?i)，在 compile 時統一指定 IGNORECASE） ----------
_SQL_INJECTION_PATTERNS = [
    # 常見 SQL 關鍵字 / 聲明（邏輯或結尾）
    r"\b(select|union|insert|update|delete|drop|truncate|alter|create|exec|execute|replace)\b",
    # SQL 注入常見短語（簡單的 or/and 判斷式）
    r"\b(or|and)\b\s+[\w'\"`]+\s*=\s*[\w'\"`]+",
    # 結尾注入：' OR '1'='1 / ' OR 1=1 等
    r"(['\"`])\s*or\s+\1?1\1?\s*=\s*\1?1\1?",
    r";\s*--",
    r"/\*.*\*/",  # block comment（跨行）
    r"\b--\b",    # inline comment
    r"benchmark\(",    # mysql expensive function abuse
    r"sleep\(",        # time-based injection
    r"information_schema",
]

_XSS_PATTERNS = [
    r"<\s*script\b",
    r"on\w+\s*=",        # onload, onclick ... attributes
    r"javascript\s*:",
    r"<\s*img\b.*\bon\w+\s*=",
    r"<\s*iframe\b",
]

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",        # ../
    r"\.\.\\",       # ..\
    r"/etc/passwd",  # linux sensitive
    r"c:\\windows",  # windows sensitive
]

# 使用非捕獲群組 (?:...) 並在 compile 時指定 IGNORECASE / DOTALL
_SQL_RE = re.compile("|".join(f"(?:{p})" for p in _SQL_INJECTION_PATTERNS),
                     re.IGNORECASE | re.DOTALL)
_XSS_RE = re.compile("|".join(f"(?:{p})" for p in _XSS_PATTERNS),
                    re.IGNORECASE | re.DOTALL)
_PATH_RE = re.compile("|".join(f"(?:{p})" for p in _PATH_TRAVERSAL_PATTERNS),
                      re.IGNORECASE)

# 檔名驗證（允許英數、底線、破折號、點、空格，長度上限 255）
_FILENAME_RE = re.compile(r"^[\w\-. ]{1,255}$", re.UNICODE)


# ---------- 檢查函數 ----------
def is_probably_sql_injection(s: str) -> Tuple[bool, Optional[str]]:
    """偵測是否含有可能的 SQL injection 模式。回傳 (bool, matched_pattern_or_None)。"""
    if not isinstance(s, str) or not s:
        return False, None
    m = _SQL_RE.search(s)
    if m:
        return True, m.group(0)
    return False, None


def is_probably_xss(s: str) -> Tuple[bool, Optional[str]]:
    """偵測是否含有可能的 XSS 模式。"""
    if not isinstance(s, str) or not s:
        return False, None
    m = _XSS_RE.search(s)
    if m:
        return True, m.group(0)
    return False, None


def contains_control_chars(s: str) -> bool:
    """檢查是否包含非列印控制字元（如 Null 等）。"""
    return bool(_CONTROL_CHARS_RE.search(s or ""))


def is_path_traversal(s: str) -> bool:
    """檢查是否包含路徑穿越或敏感系統路徑標記。"""
    if not isinstance(s, str) or not s:
        return False
    return bool(_PATH_RE.search(s))


# ---------- 型別化驗證（白名單優先） ----------
def validate_uuid(s: str) -> bool:
    """驗證是否為合法 UUID 字串（版本不限）。"""
    if not isinstance(s, str) or not s:
        return False
    try:
        UUID(s)
        return True
    except Exception:
        return False


def validate_int(s: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> bool:
    """驗證是否為整數字串且在範圍內（若指定）。"""
    if not isinstance(s, str) or not s:
        return False
    try:
        v = int(s)
    except Exception:
        return False
    if min_val is not None and v < min_val:
        return False
    if max_val is not None and v > max_val:
        return False
    return True


def validate_filename(s: str) -> bool:
    """驗證檔名（不含路徑），避免 ../ 或特殊符號。"""
    if not isinstance(s, str) or not s:
        return False
    if is_path_traversal(s):
        return False
    return bool(_FILENAME_RE.match(s))


def validate_json(s: str) -> bool:
    """檢查是否合法 JSON（且為 object/array）。"""
    if not isinstance(s, str) or not s:
        return False
    try:
        v = json.loads(s)
        return isinstance(v, (dict, list))
    except Exception:
        return False


# ---------- 綜合檢查器 ----------
def is_input_safe_for_sql(s: str, *, allow_unicode: bool = True) -> Tuple[bool, List[str]]:
    """
    綜合性的快速檢查器，回傳 (is_safe, [reasons...])
    用途：在把 user input 插入 SQL 前快速 guard（但不要取代 prepared statement）。
    """
    reasons: List[str] = []
    if s is None:
        reasons.append("input is None")
        return False, reasons
    if not isinstance(s, str):
        reasons.append("not a string")
        return False, reasons

    if not allow_unicode:
        # 若不允許 unicode，可以檢查是否為 ASCII
        try:
            s.encode("ascii")
        except UnicodeEncodeError:
            reasons.append("non-ascii characters present")

    if contains_control_chars(s):
        reasons.append("contains control characters")

    inj, pat = is_probably_sql_injection(s)
    if inj:
        # 注意：為了避免回傳過長的原始匹配內容，這裡只回傳前 200 字節
        reasons.append(f"sql-like pattern matched: {repr(pat[:200])}")

    xss, xp = is_probably_xss(s)
    if xss:
        reasons.append(f"xss-like pattern matched: {repr(xp[:200])}")

    if is_path_traversal(s):
        reasons.append("path traversal pattern")

    # 如果理由為空，表示通過快速檢查
    return (len(reasons) == 0), reasons


# ---------- 顯示/輸出安全化工具 ----------
def sanitize_for_html(s: Optional[str]) -> str:
    """在要回傳給前端頁面時做 HTML escape（避免 XSS）。"""
    if s is None:
        return ""
    return html.escape(s)


# ---------- 單元測試 / 範例（可直接執行檔案作快速檢測） ----------
if __name__ == "__main__":
    tests = [
        "normal text",
        "1; DROP TABLE users; --",
        "Robert'); DROP TABLE Students;--",
        "<script>alert(1)</script>",
        "../secrets.txt",
        "31517165-19e5-44d5-8562-350cb071d1ae",
        "select * from users where id = 1",
        "OR '1'='1'",
        "nice_file-name.txt",
        "somefile; sleep(10);",
    ]
    for t in tests:
        safe, reasons = is_input_safe_for_sql(t)
        print(f"INPUT: {t!r}")
        if safe:
            print("  => SAFE")
        else:
            print(f"  => UNSAFE: {reasons}")
        print("-" * 60)
