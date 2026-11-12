"""Async SQLite access (aiosqlite) with a simple versioned schema.

One shared connection in WAL mode serves the whole app. A lock keeps statements
from different requests from interleaving inside a transaction. SQLite is
meant for a single server process; for many replicas move to a client/server database.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence

import aiosqlite

log = logging.getLogger("psai.db")

SCHEMA_VERSION = 1
SCHEMA_V1 = """
CREATE TABLE users (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_seen  REAL NOT NULL
);

CREATE TABLE conversations (
    pk         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    public_id  TEXT NOT NULL,
    title      TEXT NOT NULL,
    subject    TEXT,
    level      TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (user_id, public_id)
);
CREATE INDEX idx_conversations_user_updated ON conversations (user_id, updated_at DESC);

CREATE TABLE messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_pk INTEGER NOT NULL REFERENCES conversations(pk) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    model           TEXT,
    provider        TEXT,
    meta            TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL
);
CREATE INDEX idx_messages_conversation ON messages (conversation_pk, id);

CREATE TABLE preferences (
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE memories (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind                TEXT NOT NULL DEFAULT 'fact',
    content             TEXT NOT NULL,
    importance          REAL NOT NULL DEFAULT 0.5,
    source_conversation TEXT,
    created_at          REAL NOT NULL
);
CREATE INDEX idx_memories_user ON memories (user_id, created_at DESC);
"""
MIGRATIONS = {1: SCHEMA_V1}  # add 2: "ALTER TABLE ..." here for future schema changes


@dataclass
class ExecResult:
    lastrowid: Optional[int]
    rowcount: int


class _Tx:
    """Statements inside `async with db.transaction() as tx` (the lock is already held)."""

    def __init__(self, conn: aiosqlite.Connection):
        self._c = conn

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> ExecResult:
        cur = await self._c.execute(sql, params)
        result = ExecResult(cur.lastrowid, cur.rowcount)
        await cur.close()
        return result

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[dict[str, Any]]:
        cur = await self._c.execute(sql, params)
        row = await cur.fetchone()
        await cur.close()
        return dict(row) if row else None

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = await self._c.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()
        return [dict(r) for r in rows]


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path, isolation_level=None)  # we manage transactions explicitly
        self._conn.row_factory = aiosqlite.Row
        for pragma in ("journal_mode=WAL", "foreign_keys=ON", "synchronous=NORMAL", "busy_timeout=5000"):
            await self._conn.execute(f"PRAGMA {pragma}")
        await self._migrate()

    async def _migrate(self) -> None:
        assert self._conn is not None
        cur = await self._conn.execute("PRAGMA user_version")
        current = (await cur.fetchone())[0]
        await cur.close()
        for version in range(current + 1, SCHEMA_VERSION + 1):
            log.info("Applying database migration %d", version)
            await self._conn.executescript(MIGRATIONS[version])
            await self._conn.execute(f"PRAGMA user_version = {version}")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def ping(self) -> bool:
        try:
            async with self._lock:
                assert self._conn is not None
                cur = await self._conn.execute("SELECT 1")
                await cur.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_Tx]:
        assert self._conn is not None, "Database is not connected"
        async with self._lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield _Tx(self._conn)
            except BaseException:
                await self._conn.rollback()
                raise
            else:
                await self._conn.commit()

    # Convenience wrappers: each call is its own transaction.
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> ExecResult:
        async with self.transaction() as tx:
            return await tx.execute(sql, params)

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[dict[str, Any]]:
        async with self._lock:
            return await _Tx(self._conn).fetchone(sql, params)  # type: ignore[arg-type]

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with self._lock:
            return await _Tx(self._conn).fetchall(sql, params)  # type: ignore[arg-type]
