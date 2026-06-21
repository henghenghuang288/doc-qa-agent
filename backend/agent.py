"""
RAG 问答 Agent

流程:
  1. 用户提问 -> Agent 调用 search_documents 工具,在该会话已上传的文档里做 BM25 检索
  2. 工具返回命中的原文片段(带来源文件名)
  3. Agent 必须只依据这些片段作答,片段里没有的内容要明确说"未找到",不能用自身知识编造
  4. 返回结果里附带引用的原文片段,前端会把它和答案一起展示,方便核对答案是否真的来自文档

三种模式,按可用的 API Key 自动切换:
  - 配置 DEEPSEEK_API_KEY  -> DeepSeek 模式(走 OpenAI 兼容接口,支付宝充值,中文好,性价比高)
  - 配置 ANTHROPIC_API_KEY -> Claude 模式
  - 都没有                 -> 纯检索模式:直接返回 BM25 命中的原文片段,不做 LLM 总结
                            (这个模式本身是诚实且有用的,很多企业内部搜索工具本质就是这个)

DeepSeek 与 OpenAI 的 Tool Calling 格式一致,因此用 openai SDK 调用,只需把 base_url 指向 DeepSeek。
"""

import json
import os
from typing import Any

from .retrieval import BM25Index

# OpenAI / DeepSeek 风格的工具定义
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在用户已上传的文档中检索与问题相关的原文片段,返回最相关的若干段落及其来源文件名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题本身"},
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "你是一个企业文档问答助手。用户会基于已上传的文档提问,你必须先调用 search_documents 工具检索相关原文片段,"
    "然后只依据检索到的片段内容回答问题。如果检索结果与问题无关,或片段中找不到答案,"
    "必须明确告知用户'未能在文档中找到相关信息',绝对不能使用你自身的知识编造答案。"
    "回答要用自然、口语化的中文,像企业客服一样清晰友好。"
    "最终请只输出一个 JSON 对象(不要带 markdown 代码块标记),字段为: "
    "answer(字符串,你给用户的回答), grounded(布尔值,是否基于文档作答), "
    "cited_chunk_ids(数组,你引用的 chunk_id 列表)。"
)


def answer_offline(index: BM25Index, question: str) -> dict[str, Any]:
    """无 API Key 时的纯检索模式:直接返回命中的原文片段,不做LLM总结。"""
    results = index.search(question, top_k=4)
    trace = [{"tool": "search_documents", "input": {"query": question}, "output_count": len(results)}]

    if not results:
        return {
            "mode": "offline_retrieval",
            "answer": "未能在已上传文档中检索到相关内容(纯检索模式不做语义改写,建议换个关键词,或配置 API Key 切换到智能问答模式)。",
            "grounded": False,
            "citations": [],
            "trace": trace,
        }

    return {
        "mode": "offline_retrieval",
        "answer": "当前为纯检索模式(未配置 API Key):以下是命中度最高的原文片段,按相关性排序。",
        "grounded": True,
        "citations": results,
        "trace": trace,
    }


def _get_llm_config():
    """返回 (api_key, base_url, model, mode_name),决定用哪个模型。优先 DeepSeek。"""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return (os.environ["DEEPSEEK_API_KEY"], "https://api.deepseek.com", "deepseek-chat", "deepseek")
    if os.environ.get("OPENAI_API_KEY"):
        return (os.environ["OPENAI_API_KEY"], None, "gpt-4o-mini", "openai")
    return (None, None, None, None)


def answer_live_openai(index: BM25Index, question: str, api_key: str, base_url: str, model: str, mode_name: str) -> dict[str, Any]:
    """用 OpenAI 兼容接口(DeepSeek/OpenAI)做带工具调用的多轮问答。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    trace: list[dict[str, Any]] = []
    last_results: list[dict] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(4):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            max_tokens=1500,
        )
        msg = response.choices[0].message

        # 没有工具调用 -> 这是最终回答
        if not msg.tool_calls:
            final_text = msg.content or ""
            cleaned = final_text.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = {"answer": final_text, "grounded": None, "cited_chunk_ids": []}

            cited_ids = set(parsed.get("cited_chunk_ids", []))
            citations = [r for r in last_results if r["chunk_id"] in cited_ids] or last_results

            return {
                "mode": f"live_{mode_name}",
                "answer": parsed.get("answer", final_text),
                "grounded": parsed.get("grounded"),
                "citations": citations,
                "trace": trace,
            }

        # 有工具调用 -> 执行检索,把结果回填
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"query": question}
            query = args.get("query", question)
            results = index.search(query, top_k=4)
            last_results = results
            trace.append({"tool": "search_documents", "input": {"query": query}, "output_count": len(results)})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(
                    [{"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"]} for r in results],
                    ensure_ascii=False,
                ),
            })

    return {"mode": f"live_{mode_name}", "answer": "", "grounded": False, "citations": [], "trace": trace, "error": "超过最大轮次未收敛"}


def answer_live_claude(index: BM25Index, question: str) -> dict[str, Any]:
    """Claude 模式(保留,以备将来用 Anthropic key)。"""
    import anthropic

    claude_tools = [{
        "name": "search_documents",
        "description": OPENAI_TOOLS[0]["function"]["description"],
        "input_schema": OPENAI_TOOLS[0]["function"]["parameters"],
    }]
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    trace: list[dict[str, Any]] = []
    messages = [{"role": "user", "content": question}]
    last_results: list[dict] = []

    for _ in range(4):
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1500,
            system=SYSTEM_PROMPT, tools=claude_tools, messages=messages,
        )
        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            cleaned = final_text.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = {"answer": final_text, "grounded": None, "cited_chunk_ids": []}
            cited_ids = set(parsed.get("cited_chunk_ids", []))
            citations = [r for r in last_results if r["chunk_id"] in cited_ids] or last_results
            return {"mode": "live_claude", "answer": parsed.get("answer", ""),
                    "grounded": parsed.get("grounded"), "citations": citations, "trace": trace}

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            results = index.search(block.input.get("query", question), top_k=4)
            last_results = results
            trace.append({"tool": "search_documents", "input": block.input, "output_count": len(results)})
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps([{"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"]} for r in results], ensure_ascii=False),
            })
        messages.append({"role": "user", "content": tool_results})

    return {"mode": "live_claude", "answer": "", "grounded": False, "citations": [], "trace": trace, "error": "超过最大轮次未收敛"}


def answer_question(index: BM25Index, question: str) -> dict[str, Any]:
    api_key, base_url, model, mode_name = _get_llm_config()
    if api_key:
        try:
            return answer_live_openai(index, question, api_key, base_url, model, mode_name)
        except Exception as e:
            # API 出问题时降级到纯检索,并把错误带回去方便排查,避免界面整个挂掉
            fallback = answer_offline(index, question)
            fallback["note"] = f"智能模式调用失败,已降级为纯检索: {type(e).__name__}"
            return fallback

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return answer_live_claude(index, question)
        except Exception as e:
            fallback = answer_offline(index, question)
            fallback["note"] = f"Claude 调用失败,已降级为纯检索: {type(e).__name__}"
            return fallback

    return answer_offline(index, question)
