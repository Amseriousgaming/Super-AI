"""Conversation storage, user preferences and long-term memory (SQLite).

Everything is scoped to a `user_id` (the frontend's X-Client-Id). Retrieval of long-term
memories is keyword based and runs in Python; the `retrieve_memories` method is the single
place to swap in embeddings or a vector index later without touching callers.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Optional

from database import Database

_TOKEN = re.compile(r"[a-z0-9]{3,}")
MAX_CONTEXT_CHARS = 1800


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def make_title(text: str, has_images: bool = False) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return "Image question" if has_images else "New conversation"
    return clean if len(clean) <= 60 else clean[:59].rstrip() + "\u2026"


class MemoryService:
    def __init__(self, db: Database, settings):
        self.db, self.settings = db, settings

    # --- users -----------------------------------------------------------------
    @staticmethod
    async def _touch_user(tx, user_id: str) -> None:
        now = time.time()
        await tx.execute(
            "INSERT INTO users (id, created_at, last_seen) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen", (user_id, now, now))

    # --- conversations -----------------------------------------------------------
    async def load_history(self, user_id: str, public_id: str, limit: int) -> list[tuple[str, str]]:
        rows = await self.db.fetchall(
            "SELECT m.role, m.content FROM messages m JOIN conversations c ON c.pk = m.conversation_pk "
            "WHERE c.user_id = ? AND c.public_id = ? ORDER BY m.id DESC LIMIT ?", (user_id, public_id, max(limit, 0)))
        return [(r["role"], r["content"]) for r in reversed(rows)]

    async def save_turn(self, user_id: str, public_id: str, question: str, answer: str, *, model: str, provider: str,
                        subject: str, level: str, meta: dict[str, Any], image_count: int = 0) -> None:
        now = time.time()
        max_chars = 100_000
        async with self.db.transaction() as tx:
            await self._touch_user(tx, user_id)
            row = await tx.fetchone("SELECT pk FROM conversations WHERE user_id = ? AND public_id = ?", (user_id, public_id))
            if row is None:
                res = await tx.execute(
                    "INSERT INTO conversations (user_id, public_id, title, subject, level, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, public_id, make_title(question, image_count > 0), subject, level, now, now))
                pk = res.lastrowid
            else:
                pk = row["pk"]
                # A retry or regenerate repeats the same question: replace the previous turn instead of duplicating it.
                last = await tx.fetchall("SELECT id, role, content FROM messages WHERE conversation_pk = ? "
                                         "ORDER BY id DESC LIMIT 2", (pk,))
                if len(last) == 2 and last[0]["role"] == "assistant" and last[1]["role"] == "user" \
                        and last[1]["content"] == question[:max_chars]:
                    await tx.execute("DELETE FROM messages WHERE id IN (?, ?)", (last[0]["id"], last[1]["id"]))
                await tx.execute("UPDATE conversations SET updated_at = ?, subject = ?, level = ? WHERE pk = ?",
                                 (now, subject, level, pk))
            await tx.execute("INSERT INTO messages (conversation_pk, role, content, meta, created_at) VALUES (?, 'user', ?, ?, ?)",
                             (pk, question[:max_chars], json.dumps({"images": image_count}), now))
            await tx.execute(
                "INSERT INTO messages (conversation_pk, role, content, model, provider, meta, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?)",
                (pk, answer[:max_chars], model, provider, json.dumps(meta, default=str), now + 0.001))

    async def list_conversations(self, user_id: str, limit: int, offset: int) -> tuple[list[dict], int]:
        rows = await self.db.fetchall(
            "SELECT c.public_id AS id, c.title, c.subject, c.level, c.created_at, c.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.conversation_pk = c.pk) AS message_count "
            "FROM conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset))
        total = await self.db.fetchone("SELECT COUNT(*) AS n FROM conversations WHERE user_id = ?", (user_id,))
        return rows, (total or {"n": 0})["n"]

    async def get_conversation(self, user_id: str, public_id: str, max_messages: int = 500) -> Optional[dict]:
        conv = await self.db.fetchone(
            "SELECT pk, public_id AS id, title, subject, level, created_at, updated_at FROM conversations "
            "WHERE user_id = ? AND public_id = ?", (user_id, public_id))
        if conv is None:
            return None
        msgs = await self.db.fetchall(
            "SELECT id, role, content, model, provider, created_at, meta FROM messages WHERE conversation_pk = ? "
            "ORDER BY id LIMIT ?", (conv["pk"], max_messages))
        for m in msgs:
            m["meta"] = json.loads(m["meta"] or "{}")
        conv["message_count"] = len(msgs)
        conv["messages"] = msgs
        conv.pop("pk")
        return conv

    async def delete_conversation(self, user_id: str, public_id: str) -> bool:
        res = await self.db.execute("DELETE FROM conversations WHERE user_id = ? AND public_id = ?", (user_id, public_id))
        return res.rowcount > 0

    # --- preferences -------------------------------------------------------------
    async def get_preferences(self, user_id: str) -> dict[str, Any]:
        rows = await self.db.fetchall("SELECT key, value FROM preferences WHERE user_id = ?", (user_id,))
        return {r["key"]: json.loads(r["value"]) for r in rows}

    async def set_preferences(self, user_id: str, prefs: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        async with self.db.transaction() as tx:
            await self._touch_user(tx, user_id)
            for key, value in prefs.items():
                if value is None:
                    await tx.execute("DELETE FROM preferences WHERE user_id = ? AND key = ?", (user_id, key))
                else:
                    await tx.execute(
                        "INSERT INTO preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                        (user_id, key, json.dumps(value), now))
        return await self.get_preferences(user_id)

    # --- long-term memory --------------------------------------------------------
    async def add_memory(self, user_id: str, content: str, kind: str = "fact", importance: float = 0.5,
                         source_conversation: Optional[str] = None) -> int:
        kind = kind if kind in {"fact", "preference", "goal"} else "fact"
        async with self.db.transaction() as tx:
            await self._touch_user(tx, user_id)
            dup = await tx.fetchone("SELECT id FROM memories WHERE user_id = ? AND lower(content) = lower(?)", (user_id, content))
            if dup:
                return dup["id"]
            res = await tx.execute(
                "INSERT INTO memories (user_id, kind, content, importance, source_conversation, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (user_id, kind, content[:500], importance, source_conversation, time.time()))
            await tx.execute(  # keep the store bounded: drop the least important, oldest entries
                "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE user_id = ? "
                "ORDER BY importance ASC, created_at ASC LIMIT max(0, (SELECT COUNT(*) FROM memories WHERE user_id = ?) - ?))",
                (user_id, user_id, self.settings.max_memories_per_user))
            return res.lastrowid or 0

    async def list_memories(self, user_id: str, limit: int = 200) -> list[dict]:
        return await self.db.fetchall(
            "SELECT id, kind, content, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit))

    async def delete_memory(self, user_id: str, memory_id: int) -> bool:
        res = await self.db.execute("DELETE FROM memories WHERE user_id = ? AND id = ?", (user_id, memory_id))
        return res.rowcount > 0

    async def retrieve_memories(self, user_id: str, query: str, k: int = 5) -> list[dict]:
        """Keyword-overlap retrieval. Replace the body with embedding search when you outgrow it."""
        rows = await self.db.fetchall(
            "SELECT id, kind, content, importance, created_at FROM memories WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT 500", (user_id,))
        q = _tokens(query)
        scored = []
        for r in rows:
            overlap = len(q & _tokens(r["content"]))
            if overlap:
                scored.append((overlap / math.sqrt(len(_tokens(r["content"])) or 1) * (0.5 + r["importance"]), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:k]]

    # --- context for the prompt ---------------------------------------------------
    async def build_context(self, user_id: str, question: str, include_memories: bool) -> str:
        lines: list[str] = []
        prefs = await self.get_preferences(user_id)
        for key, value in list(prefs.items())[:20]:
            lines.append(f"- preference {key}: {value}")
        if include_memories:
            for m in await self.retrieve_memories(user_id, question):
                lines.append(f"- {m['kind']}: {m['content']}")
        return "\n".join(lines)[:MAX_CONTEXT_CHARS]
