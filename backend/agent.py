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

全程使用异步客户端(AsyncOpenAI/AsyncAnthropic):FastAPI 服务一次要同时处理多个用户的提问请求,
如果用同步客户端,等大模型 API 返回的这几秒里整个进程会被卡住、没法处理别的请求;
异步客户端在等待网络 I/O 时能让出控制权去处理其他请求,这是生产环境 LLM 应用的标准做法。

同时记录每次调用的耗时和 token 消耗(cost/latency 可观测性最基础的一步):
每次大模型调用都会在 trace 里附带 latency_ms 和 usage(prompt/completion/total token 数),
这是后续做成本治理、限流、按任务分级选型的数据基础。
"""

import ast
import json
import operator
import os
import time
from datetime import datetime
from typing import Any

from .retrieval import BM25Index

# 记录答不上来的问题——这是"产品反馈闭环"的起点：
# 收集真实用户问到但文档里没有答案的问题，定期补充进文档，系统越用越聪明。
_UNANSWERABLE_LOG: list[dict] = []

def _log_unanswerable(question: str) -> None:
    _UNANSWERABLE_LOG.append({
        "question": question,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    # 只保留最近100条，防止内存无限增长
    if len(_UNANSWERABLE_LOG) > 100:
        _UNANSWERABLE_LOG.pop(0)

def get_unanswerable_log() -> list[dict]:
    return list(_UNANSWERABLE_LOG)

# 安全计算:只允许数值运算(加减乘除、乘方、取余、括号、正负号),
# 不能用 eval() 直接执行字符串——那样等于让模型输出的任意文本在服务器上跑代码,是真实的安全隐患。
# 用 ast 解析表达式,只认白名单里的运算符,遇到其他东西(比如函数调用、变量名)直接拒绝。
_SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"表达式包含不允许的内容: {ast.dump(node)}")


def safe_calculate(expression: str) -> dict[str, Any]:
    """安全计算一个纯数值表达式,出错时返回错误信息而不是抛异常崩掉整个请求。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree.body)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"无法计算这个表达式: {type(e).__name__}"}

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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个纯数值表达式(加减乘除、乘方、括号),用于回答涉及计算的问题,"
                            "例如根据文档中查到的单价算总价。不要自己心算,数字计算一律调用这个工具,避免算错。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "纯数值表达式,例如 '899 * 12'"},
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你是一个企业文档问答助手。你有两个工具:search_documents 用来在已上传文档中检索相关原文片段,"
    "calculate 用来做数值计算。用户提问时,先判断需要哪个或哪几个工具——"
    "如果问题需要先从文档里查到一个数字、再做计算(比如'订一年多少钱'要先查到月费再乘以12),"
    "就先调用 search_documents 拿到数字,再调用 calculate 算出结果,两个工具可以在一次回答里依次使用。"
    "数字计算永远调用 calculate,不要自己心算,大模型心算数字容易出错。"
    "你必须只依据 search_documents 检索到的片段内容回答事实性问题,如果片段里找不到答案,"
    "必须明确告知用户'未能在文档中找到相关信息',绝对不能使用你自身的知识编造答案。"
    "回答要用自然、口语化的中文,像企业客服一样清晰友好。"
    "最终请只输出一个 JSON 对象(不要带 markdown 代码块标记),字段为: "
    "answer(字符串,你给用户的回答), grounded(布尔值,是否基于文档作答), "
    "cited_chunk_ids(数组,你引用的 chunk_id 列表)。"
)


def answer_offline(index: BM25Index, question: str) -> dict[str, Any]:
    """无 API Key 时的纯检索模式:直接返回命中的原文片段,不做LLM总结。"""
    t0 = time.perf_counter()
    results = index.search(question, top_k=4)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace = [{"tool": "search_documents", "input": {"query": question}, "output_count": len(results), "latency_ms": latency_ms}]

    if not results:
        return {
            "mode": "offline_retrieval",
            "answer": "未能在已上传文档中检索到相关内容(纯检索模式不做语义改写,建议换个关键词,或配置 API Key 切换到智能问答模式)。",
            "grounded": False,
            "citations": [],
            "trace": trace,
            "usage": None,
        }

    return {
        "mode": "offline_retrieval",
        "answer": "当前为纯检索模式(未配置 API Key):以下是命中度最高的原文片段,按相关性排序。",
        "grounded": True,
        "citations": results,
        "trace": trace,
        "usage": None,
    }


def _get_llm_config():
    """返回 (api_key, base_url, model, mode_name),决定用哪个模型。优先 DeepSeek。"""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return (os.environ["DEEPSEEK_API_KEY"], "https://api.deepseek.com", "deepseek-chat", "deepseek")
    if os.environ.get("OPENAI_API_KEY"):
        # 支持自定义 base_url:客户内网自建的开源模型(vLLM 等)只要兼容 OpenAI 接口即可接入,
        # 实现"数据零外发"的完全本地化部署。未设置则走 OpenAI 官方。
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        mode = "local_model" if base_url else "openai"
        return (os.environ["OPENAI_API_KEY"], base_url, model, mode)
    return (None, None, None, None)


def _usage_dict(response) -> dict | None:
    """从 API 响应里取出 token 用量,统一成 dict,取不到就返回 None 而不是报错。"""
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _merge_usage(total: dict | None, new: dict | None) -> dict | None:
    """多轮工具调用会产生多次 API 调用,把每轮的 token 消耗累加成本次提问的总消耗。"""
    if new is None:
        return total
    if total is None:
        return dict(new)
    return {k: (total.get(k) or 0) + (new.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


async def answer_live_openai(index: BM25Index, question: str, api_key: str, base_url: str, model: str, mode_name: str) -> dict[str, Any]:
    """用 OpenAI 兼容接口(DeepSeek/OpenAI)做带工具调用的多轮问答,异步客户端,不阻塞其他并发请求。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
    trace: list[dict[str, Any]] = []
    last_results: list[dict] = []
    total_usage: dict | None = None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(4):
        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            max_tokens=1500,
        )
        call_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        call_usage = _usage_dict(response)
        total_usage = _merge_usage(total_usage, call_usage)
        msg = response.choices[0].message

        # 没有工具调用 -> 这是最终回答
        if not msg.tool_calls:
            final_text = msg.content or ""
            cleaned = final_text.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # JSON解析失败时，检查是否像一个残缺的JSON（以{开头）
                # 如果是，说明模型输出格式有问题，提取可读部分或给友好提示
                # 不能把原始JSON残文直接暴露给用户
                if cleaned.startswith("{"):
                    # 尝试提取answer字段的值
                    import re
                    m = re.search(r'"answer"\s*:\s*"(.*?)"(?:,|\})', cleaned, re.DOTALL)
                    answer_text = m.group(1) if m else "抱歉，回答生成时出现格式问题，请重新提问。"
                    parsed = {"answer": answer_text, "grounded": None, "cited_chunk_ids": []}
                else:
                    parsed = {"answer": final_text, "grounded": None, "cited_chunk_ids": []}

            cited_ids = set(parsed.get("cited_chunk_ids", []))
            citations = [r for r in last_results if r["chunk_id"] in cited_ids] or last_results

            answer = parsed.get("answer", "")
            grounded = parsed.get("grounded")

            # 转人工逻辑：答案明确表示未找到时，附加转人工提示
            UNANSWERABLE_SIGNALS = ["未能在文档", "未找到", "没有找到", "无法回答", "文档中没有", "找不到相关"]
            needs_escalation = (
                grounded is False or
                any(s in answer for s in UNANSWERABLE_SIGNALS)
            )
            if needs_escalation and not citations:
                escalation_hint = "\n\n💬 此问题超出文档范围，建议转人工客服处理。"
                answer = answer + escalation_hint
                _log_unanswerable(question)

            trace.append({"event": "llm_final_answer", "latency_ms": call_latency_ms, "usage": call_usage})

            return {
                "mode": f"live_{mode_name}",
                "answer": answer,
                "grounded": grounded,
                "citations": citations,
                "needs_escalation": needs_escalation and not citations,
                "trace": trace,
                "usage": total_usage,
            }

        # 有工具调用 -> 执行检索,把结果回填
        trace.append({"event": "llm_tool_call_decision", "latency_ms": call_latency_ms, "usage": call_usage})
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
                args = {}

            if tc.function.name == "calculate":
                t_calc = time.perf_counter()
                calc_result = safe_calculate(args.get("expression", ""))
                calc_latency_ms = round((time.perf_counter() - t_calc) * 1000, 1)
                trace.append({"tool": "calculate", "input": args, "output": calc_result, "latency_ms": calc_latency_ms})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(calc_result, ensure_ascii=False),
                })
                continue

            query = args.get("query", question)
            t_search = time.perf_counter()
            results = index.search(query, top_k=4)
            search_latency_ms = round((time.perf_counter() - t_search) * 1000, 1)
            last_results = results
            trace.append({"tool": "search_documents", "input": {"query": query}, "output_count": len(results), "latency_ms": search_latency_ms})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(
                    [{"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"]} for r in results],
                    ensure_ascii=False,
                ),
            })

    return {"mode": f"live_{mode_name}", "answer": "", "grounded": False, "citations": [], "trace": trace, "usage": total_usage, "error": "超过最大轮次未收敛"}


async def answer_live_claude(index: BM25Index, question: str) -> dict[str, Any]:
    """Claude 模式(保留,以备将来用 Anthropic key),异步客户端。"""
    import anthropic

    claude_tools = [
        {
            "name": "search_documents",
            "description": OPENAI_TOOLS[0]["function"]["description"],
            "input_schema": OPENAI_TOOLS[0]["function"]["parameters"],
        },
        {
            "name": "calculate",
            "description": OPENAI_TOOLS[1]["function"]["description"],
            "input_schema": OPENAI_TOOLS[1]["function"]["parameters"],
        },
    ]
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    trace: list[dict[str, Any]] = []
    messages = [{"role": "user", "content": question}]
    last_results: list[dict] = []
    total_usage: dict | None = None

    for _ in range(4):
        t0 = time.perf_counter()
        response = await client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1500,
            system=SYSTEM_PROMPT, tools=claude_tools, messages=messages,
        )
        call_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        usage = getattr(response, "usage", None)
        call_usage = None
        if usage:
            call_usage = {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
            }
        total_usage = _merge_usage(total_usage, call_usage)

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            cleaned = final_text.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = {"answer": final_text, "grounded": None, "cited_chunk_ids": []}
            cited_ids = set(parsed.get("cited_chunk_ids", []))
            citations = [r for r in last_results if r["chunk_id"] in cited_ids] or last_results
            trace.append({"event": "llm_final_answer", "latency_ms": call_latency_ms, "usage": call_usage})
            return {"mode": "live_claude", "answer": parsed.get("answer", ""),
                    "grounded": parsed.get("grounded"), "citations": citations, "trace": trace, "usage": total_usage}

        trace.append({"event": "llm_tool_call_decision", "latency_ms": call_latency_ms, "usage": call_usage})
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "calculate":
                t_calc = time.perf_counter()
                calc_result = safe_calculate(block.input.get("expression", ""))
                calc_latency_ms = round((time.perf_counter() - t_calc) * 1000, 1)
                trace.append({"tool": "calculate", "input": block.input, "output": calc_result, "latency_ms": calc_latency_ms})
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(calc_result, ensure_ascii=False),
                })
                continue

            t_search = time.perf_counter()
            results = index.search(block.input.get("query", question), top_k=4)
            search_latency_ms = round((time.perf_counter() - t_search) * 1000, 1)
            last_results = results
            trace.append({"tool": "search_documents", "input": block.input, "output_count": len(results), "latency_ms": search_latency_ms})
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps([{"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"]} for r in results], ensure_ascii=False),
            })
        messages.append({"role": "user", "content": tool_results})

    return {"mode": "live_claude", "answer": "", "grounded": False, "citations": [], "trace": trace, "usage": total_usage, "error": "超过最大轮次未收敛"}


async def answer_question(index: BM25Index, question: str) -> dict[str, Any]:
    request_t0 = time.perf_counter()
    api_key, base_url, model, mode_name = _get_llm_config()

    if api_key:
        try:
            result = await answer_live_openai(index, question, api_key, base_url, model, mode_name)
        except Exception as e:
            # API 出问题时降级到纯检索,并把错误带回去方便排查,避免界面整个挂掉
            result = answer_offline(index, question)
            result["note"] = f"智能模式调用失败,已降级为纯检索: {type(e).__name__}"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        try:
            result = await answer_live_claude(index, question)
        except Exception as e:
            result = answer_offline(index, question)
            result["note"] = f"Claude 调用失败,已降级为纯检索: {type(e).__name__}"
    else:
        result = answer_offline(index, question)

    result["total_latency_ms"] = round((time.perf_counter() - request_t0) * 1000, 1)
    return result
