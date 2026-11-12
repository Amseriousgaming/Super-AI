"""SQLite conversation storage, preferences and long-term memory."""
import pytest

from database import Database
from services.knowledge import ArxivSource, CrossrefSource
from services.memory import MemoryService, make_title
from tests.fakes import make_settings, run

U1, U2 = "user-one-1234", "user-two-5678"


def service(**kw):
    async def build():
        db = Database(":memory:")
        await db.connect()
        return MemoryService(db, make_settings(**kw))

    return run_with_loop(build)


# One event loop for the whole test: aiosqlite connections belong to the loop that created them.
def run_with_loop(factory):
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    svc = loop.run_until_complete(factory())
    svc._loop = loop
    return svc


def go(svc, coro):
    return svc._loop.run_until_complete(coro)


def test_save_list_get_delete_and_isolation():
    m = service()
    go(m, m.save_turn(U1, "conv-0001", "What is entropy?", "A measure of disorder.", model="m", provider="p",
                      subject="physics", level="undergraduate", meta={"mode": "detailed"}))
    go(m, m.save_turn(U1, "conv-0001", "Give an example", "Ice melting.", model="m", provider="p",
                      subject="physics", level="undergraduate", meta={}))
    rows, total = go(m, m.list_conversations(U1, 10, 0))
    assert total == 1 and rows[0]["message_count"] == 4 and rows[0]["title"] == "What is entropy?"
    detail = go(m, m.get_conversation(U1, "conv-0001"))
    assert [x["role"] for x in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert go(m, m.load_history(U1, "conv-0001", 3))[0] == ("assistant", "A measure of disorder.")
    assert go(m, m.get_conversation(U2, "conv-0001")) is None          # another user cannot read it
    assert go(m, m.delete_conversation(U2, "conv-0001")) is False
    assert go(m, m.delete_conversation(U1, "conv-0001")) is True
    assert go(m, m.list_conversations(U1, 10, 0))[1] == 0


def test_repeating_a_question_replaces_the_last_turn():
    m = service()
    for answer in ("first", "second"):
        go(m, m.save_turn(U1, "conv-0002", "Same question", answer, model="m", provider="p", subject="a", level="b", meta={}))
    msgs = go(m, m.get_conversation(U1, "conv-0002"))["messages"]
    assert [x["content"] for x in msgs] == ["Same question", "second"]


def test_preferences_upsert_and_delete():
    m = service()
    assert go(m, m.set_preferences(U1, {"units": "metric", "exam": "IB"})) == {"units": "metric", "exam": "IB"}
    assert go(m, m.set_preferences(U1, {"units": None, "level_goal": 3})) == {"exam": "IB", "level_goal": 3}
    assert go(m, m.get_preferences(U2)) == {}


def test_long_term_memory_retrieval_dedupe_and_cap():
    m = service(max_memories_per_user=10)
    go(m, m.add_memory(U1, "Studying organic chemistry for the MCAT", kind="goal"))
    go(m, m.add_memory(U1, "studying organic chemistry for the mcat"))  # duplicate (case-insensitive)
    go(m, m.add_memory(U1, "Prefers worked examples over theory"))
    assert len(go(m, m.list_memories(U1))) == 2
    hits = go(m, m.retrieve_memories(U1, "help me with chemistry reactions"))
    assert hits and "organic chemistry" in hits[0]["content"]
    assert go(m, m.retrieve_memories(U2, "chemistry")) == []
    for i in range(15):
        go(m, m.add_memory(U1, f"fact number {i} about topic{i}"))
    assert len(go(m, m.list_memories(U1))) == 10
    # the oldest entries were evicted by the cap; the newest are still retrievable
    ctx = go(m, m.build_context(U1, "tell me about topic14", include_memories=True))
    assert "fact number 14" in ctx
    assert go(m, m.build_context(U1, "tell me about topic14", include_memories=False)) == ""


def test_titles():
    assert make_title("  hello   world ") == "hello world"
    assert make_title("", has_images=True) == "Image question"
    assert len(make_title("x" * 200)) == 60


ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1706.03762v7</id>
<title>Attention Is   All You Need</title><summary>The dominant sequence transduction models...</summary>
<published>2017-06-12T17:57:34Z</published><author><name>Ashish Vaswani</name></author></entry></feed>"""


def test_knowledge_parsers_only_report_what_the_api_returned():
    arxiv = ArxivSource.parse(ARXIV)
    assert arxiv[0].title == "Attention Is All You Need" and arxiv[0].kind == "preprint" and "not peer reviewed" in arxiv[0].note
    assert ArxivSource.parse("<not xml") == []
    data = {"message": {"items": [{"DOI": "10.1000/xyz", "title": ["A study"], "type": "journal-article",
                                   "author": [{"given": "Ada", "family": "Lovelace"}], "issued": {"date-parts": [[1843]]},
                                   "container-title": ["Notes"]}, {"title": ["No DOI, skipped"]}]}}
    crossref = CrossrefSource.parse(data)
    assert len(crossref) == 1 and crossref[0].url == "https://doi.org/10.1000/xyz" and crossref[0].kind == "journal"
