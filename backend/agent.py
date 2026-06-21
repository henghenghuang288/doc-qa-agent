"""
RAG 问答 Agent

流程:
  1. 用户提问 -> Agent 调用 search_documents 工具,在该会话已上传的文档里做 BM25 检索
  2. 工具返回命中的原文片段(带来源文件名)
  3. Agent 必须只依据这些片段作答,片段里没有的内容要明确说"未找到",不能用自身知识编造
  4. 返回结果里附带引用的原文片段,前端会把它和答案一起展示,方便核对答案是否真的来自文档

没有配置 ANTHROPIC_API_KEY 时降级为"纯检索模式":
直接返回 BM25 命中的原文片段,不做 LLM 总结。
这个模式本身是诚实且有用的(很多企业内部搜索工具本质就是这个),
而不是伪造一个假的"智能回答"。
"""

import json
import os
from typing import Any

from .retrieval import BM25Index

TOOLS = [
    {
        "name": "search_documents",
        "description": "在用户已上传的文档中检索与问题相关的原文片段,返回最相关的若干段落及其来源文件名。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题本身"},
            },
            "required": ["query"],
        },
    }
]

SYSTEM_PROMPT = (
    "你是一个企业文档问答助手。用户会基于已上传的文档提问,你必须先调用 search_documents 工具检索相关原文片段,"
    "然后只依据检索到的片段内容回答问题。如果检索结果与问题无关,或片段中找不到答案,"
    "必须明确告知用户'未能在文档中找到相关信息',绝对不能使用你自身的知识编造答案。"
    "回答末尾请用 JSON 格式给出引用了哪些 chunk_id。"
    "最终请只输出一个 JSON 对象,字段为: answer(字符串,你的回答), grounded(布尔值,是否基于文档作答),"
    "cited_chunk_ids(数组,引用的chunk_id列表)。"
)


def answer_offline(index: BM25Index, question: str) -> dict[str, Any]:
    """无 API Key 时的纯检索模式:直接返回命中的原文片段,不做LLM总结。"""
    results = index.search(question, top_k=4)
    trace = [{"tool": "search_documents", "input": {"query": question}, "output_count": len(results)}]

    if not results:
        return {
            "mode": "offline_retrieval",
            "answer": "未能在已上传文档中检索到相关内容(纯检索模式不做语义改写,建议换个关键词试试,或配置 API Key 切换到 Agent 总结模式)。",
            "grounded": False,
            "citations": [],
            "trace": trace,
        }

    return {
        "mode": "offline_retrieval",
        "answer": "未配置 ANTHROPIC_API_KEY,当前为纯检索模式:以下是命中度最高的原文片段,按相关性排序。",
        "grounded": True,
        "citations": results,
        "trace": trace,
    }


def answer_live(index: BM25Index, question: str) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    trace: list[dict[str, Any]] = []
    messages = [{"role": "user", "content": question}]
    last_results: list[dict] = []

    for _ in range(4):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            try:
                parsed = json.loads(final_text)
            except json.JSONDecodeError:
                parsed = {"answer": final_text, "grounded": None, "cited_chunk_ids": []}

            cited_ids = set(parsed.get("cited_chunk_ids", []))
            citations = [r for r in last_results if r["chunk_id"] in cited_ids] or last_results

            return {
                "mode": "live_agent",
                "answer": parsed.get("answer", ""),
                "grounded": parsed.get("grounded"),
                "citations": citations,
                "trace": trace,
            }

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            results = index.search(block.input.get("query", question), top_k=4)
            last_results = results
            trace.append({"tool": "search_documents", "input": block.input, "output_count": len(results)})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(
                    [{"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"]} for r in results],
                    ensure_ascii=False,
                ),
            })
        messages.append({"role": "user", "content": tool_results})

    return {"mode": "live_agent", "answer": "", "grounded": False, "citations": [], "trace": trace, "error": "超过最大轮次未收敛"}


def answer_question(index: BM25Index, question: str) -> dict[str, Any]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return answer_offline(index, question)
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return answer_offline(index, question)
    return answer_live(index, question)
