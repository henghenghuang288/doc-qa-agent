# Enterprise Document QA Agent

> 🇨🇳 [中文版见下方](#中文说明)

A privately-deployable enterprise document Q&A system. Upload your company documents (manuals, FAQs, policies), and the AI answers questions based **strictly** on the document content — citing the exact source passage for every answer, and honestly refusing when the answer isn't in the documents.

**Live demo:** https://doc-qa-agent-2f7j.onrender.com

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Hand-written BM25 + jieba (no LangChain) | Understand and control every layer; easier to debug and explain |
| Native Tool Calling loop | Model decides when to search vs. calculate; no framework magic |
| Safe AST calculator tool | LLMs miscalculate; let code handle math via whitelist-only AST eval |
| Anti-hallucination by design | Answer forced to ground in retrieved chunks; refusal when not found |
| Three-tier private deployment | Fully offline / DeepSeek cloud / self-hosted model on customer intranet |
| Async throughout (AsyncOpenAI) | Non-blocking; handles concurrent users without queuing |

## Evaluation System

25 golden test cases including **4 "induced fabrication" trap questions** (phrased as if the answer exists but it doesn't). Automated end-to-end scoring across the full upload→retrieve→answer pipeline.

Results on the live deployment (DeepSeek mode):
- Answerable question accuracy: **100%**
- Refusal accuracy (anti-hallucination): **88.9%**

Run evals at `/evals.html` — includes a zero-token offline mode for baseline checks.

## Private Deployment (Core Value Prop)

Unlike public AI tools (ChatGPT, DeepSeek web), this system runs entirely on the customer's own server. Three security tiers:

1. **Fully offline** — zero external requests, pure BM25 retrieval
2. **DeepSeek cloud** — document chunks sent to DeepSeek API for summarization
3. **Self-hosted model** — point `OPENAI_BASE_URL` at any OpenAI-compatible local model (e.g. vLLM + Qwen); data never leaves the intranet

## Stack

Python · FastAPI · asyncio · BM25 · jieba · DeepSeek/Claude/OpenAI-compatible · Docker

## Quick Start

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxxx   # optional; runs offline without it
uvicorn backend.main:app --reload
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/session` | Create a new session |
| `POST /api/session/{id}/upload` | Upload a document |
| `POST /api/session/{id}/ask` | Ask a question |
| `GET /api/evals[?offline=true]` | Run evaluation suite |
| `GET /api/unanswerable` | Questions the system couldn't answer (feedback loop data) |
| `GET /api/logs` | Request log (observability) |

---

## 中文说明

可私有化部署的企业文档问答系统。上传企业文档后，AI 严格基于文档原文回答，每句答案标注引用来源，文档里没有的内容拒绝作答不编造。

**核心差异化：** 系统能整体部署到客户自己的服务器上，数据不出内网——这是豆包、ChatGPT 等公网产品做不到的，解决企业数据合规问题。

**技术亮点：**
- 手写 BM25 检索 + jieba 中文分词，不依赖 LangChain 框架
- 原生 Tool Calling 多轮 Agent 循环，支持检索和计算两种工具
- 25 道金标准评估测试含诱导编造陷阱题，可回答题命中率 100%
- 全异步处理（AsyncOpenAI），支持并发请求
- 三档私有化部署方案（完全离线 / DeepSeek / 内网自建模型）
