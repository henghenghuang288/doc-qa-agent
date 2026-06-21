"""
会话存储 —— 内存态,不落盘

这是处理"真实企业文档"时一个容易被忽略但会被技术面试官追问的点:
客户文档怎么保证不被永久留存/泄露?
这里的答案是:上传内容只存在这个进程的内存里(一个 session_id 对应一份 BM25Index),
服务重启或进程退出即全部清空,没有写入磁盘数据库。
生产环境如果要做持久化,应该走加密存储 + 客户级别隔离 + 过期自动清理,
但那是另一层工程决策,MVP 阶段"不留痕"是更安全的默认值。
"""

import secrets
import time
from dataclasses import dataclass, field

from .retrieval import BM25Index

SESSION_TTL_SECONDS = 2 * 60 * 60  # 2小时未使用自动视为过期


@dataclass
class Session:
    session_id: str
    index: BM25Index = field(default_factory=BM25Index)
    filenames: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}


def create_session() -> Session:
    sid = secrets.token_urlsafe(12)
    session = Session(session_id=sid)
    _sessions[sid] = session
    return session


def get_session(session_id: str) -> Session | None:
    _cleanup_expired()
    session = _sessions.get(session_id)
    if session:
        session.last_used_at = time.time()
    return session


def _cleanup_expired() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s.last_used_at > SESSION_TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]
