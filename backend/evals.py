"""
评估运行器(Evals Runner)

跑一遍完整的真实流程:建会话 -> 上传金标准文档 -> 逐题提问 -> 对照预期结果打分。
这不是单元测试,是对"系统在真实使用路径上表现如何"的端到端评估,
对应 FDE 岗位反复强调的"golden dataset + 回归报告"。

打分规则:
  - answerable=True 的题:回答里命中了 expect_keywords 里任意一个关键词才算通过
    (不要求一字不差,因为模型会用自己的话总结,只要信息点对就行)
  - answerable=False 的题:回答被判定为"拒答"才算通过——
    判断依据是 grounded 字段为 False,或者回答文本里出现拒答类措辞

失败的题目会被完整记录下来(问题、预期、实际回答、引用片段),这份记录就是
"失败模式归因报告"的原始素材,可以直接写进 README 或讲给面试官听。
"""

import time
from typing import Any

from .agent import answer_question
from .document_processor import chunk_text
from .eval_data import GOLDEN_DOCUMENT, GOLDEN_QUESTIONS
from .retrieval import BM25Index

REFUSAL_PHRASES = ["未找到", "未能", "没有提到", "无法提供", "找不到", "未在文档", "无法找到", "资料中没有", "未明确"]


def _full_text(result: dict) -> str:
    """拼接 answer 和所有 citation 的原文,用于关键词检索——
    纯检索模式下真正的内容信息在 citations 里,answer 只是一句固定提示语,
    只查 answer 字段会把纯检索模式的所有正确结果都误判为失败。"""
    parts = [result.get("answer") or ""]
    for c in result.get("citations") or []:
        parts.append(c.get("text", ""))
    return "\n".join(parts)


def _judge_answerable_case(case: dict, result: dict) -> tuple[bool, str]:
    text = _full_text(result)
    hit = any(kw in text for kw in case["expect_keywords"])
    if hit:
        return True, "命中预期关键词(回答或引用片段中)"
    return False, f"回答与引用片段均未包含预期关键词 {case['expect_keywords']}"


def _judge_unanswerable_case(case: dict, result: dict, mode: str) -> tuple[bool, str, bool]:
    """返回 (是否通过, 原因, 是否适用该指标)。
    纯检索模式没有语义判断能力,只会把"最像"的内容甩出来,不会主动拒答——
    这是它的设计本质,不是bug,所以拒答准确率这个指标对它不适用,标记为 N/A。"""
    if mode == "offline_retrieval":
        return True, "纯检索模式不做拒答判断(设计如此),该指标不适用", False

    answer = (result.get("answer") or "")
    grounded = result.get("grounded")
    refused_by_text = any(p in answer for p in REFUSAL_PHRASES)
    refused_by_flag = grounded is False
    if refused_by_text or refused_by_flag:
        return True, "正确拒答", True
    return False, "未拒答,疑似编造了文档中不存在的内容", True


async def run_evals() -> dict[str, Any]:
    index = BM25Index()
    chunks = chunk_text(GOLDEN_DOCUMENT)
    index.add_documents(chunks, source="golden_eval_doc.txt")

    results = []
    t0 = time.time()
    detected_mode = None
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    latencies_ms = []
    for case in GOLDEN_QUESTIONS:
        r = await answer_question(index, case["question"])
        detected_mode = r.get("mode")
        if r.get("total_latency_ms") is not None:
            latencies_ms.append(r["total_latency_ms"])
        if r.get("usage"):
            for k in total_usage:
                total_usage[k] += r["usage"].get(k) or 0
        if case["answerable"]:
            passed, reason = _judge_answerable_case(case, r)
            metric_applicable = True
        else:
            passed, reason, metric_applicable = _judge_unanswerable_case(case, r, detected_mode)
        results.append({
            "id": case["id"],
            "question": case["question"],
            "answerable_expected": case["answerable"],
            "passed": passed,
            "metric_applicable": metric_applicable,
            "reason": reason,
            "answer": r.get("answer"),
            "grounded": r.get("grounded"),
            "mode": r.get("mode"),
            "latency_ms": r.get("total_latency_ms"),
            "note": case.get("note"),
        })
    elapsed = round(time.time() - t0, 1)

    scored = [r for r in results if r["metric_applicable"]]
    total = len(scored)
    passed_count = sum(1 for r in scored if r["passed"])

    answerable_cases = [r for r in scored if r["answerable_expected"]]
    unanswerable_cases = [r for r in scored if not r["answerable_expected"]]
    answerable_acc = round(100 * sum(1 for r in answerable_cases if r["passed"]) / len(answerable_cases), 1) if answerable_cases else None
    refusal_acc = round(100 * sum(1 for r in unanswerable_cases if r["passed"]) / len(unanswerable_cases), 1) if unanswerable_cases else None

    skipped_na = [r for r in results if not r["metric_applicable"]]
    failures = [r for r in scored if not r["passed"]]

    avg_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else None

    return {
        "summary": {
            "mode": detected_mode,
            "total_cases": len(results),
            "scored_cases": total,
            "na_cases": len(skipped_na),
            "passed": passed_count,
            "pass_rate": round(100 * passed_count / total, 1) if total else None,
            "answerable_accuracy": answerable_acc,
            "refusal_accuracy": refusal_acc,
            "refusal_accuracy_note": "纯检索模式下不评估拒答能力(设计上无语义判断层),需配置 API Key 切换到智能问答模式后该指标才有效" if detected_mode == "offline_retrieval" else None,
            "elapsed_seconds": elapsed,
            "avg_latency_ms": avg_latency_ms,
            "total_tokens": total_usage["total_tokens"] if any(total_usage.values()) else None,
        },
        "failures": failures,
        "results": results,
    }
