"""
检索模块 —— 手写 BM25 + jieba 中文分词

为什么不直接上向量检索:
- BM25 是关键词检索,完全可解释,不需要额外的 embedding API/费用,今天就能跑且结果可调试。
- 中文文本词与词之间没有空格,如果直接按字符或空格切词,关键词匹配基本失效,
  所以必须先用 jieba 分词,这是中文 RAG 系统里很容易被忽略但必须处理的细节。

已知局限(写在这里是为了诚实,不是为了藏起来):
BM25 本质是字面匹配,如果用户提问和文档原文用词差异较大(同义词/换种问法),召回会变差。
生产版本的升级路径是接入向量 embedding(如 Voyage AI)做语义检索,
或者 BM25 + 向量做混合检索(hybrid search),取两者并集再重排序。
这个模块的 search() 接口签名设计成不需要改动就能在未来替换底层实现。
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import jieba

jieba.setLogLevel(20)  # 关闭 jieba 初始化时的调试日志,避免污染输出


def _tokenize(text: str) -> list[str]:
    # 用搜索引擎模式而非默认模式:默认模式会把"客服热线"切成一个完整词,
    # 导致提问"客服电话"里的"客服"匹配不上。搜索引擎模式会额外保留细粒度子词,
    # 牺牲一点精度换召回率,更适合检索场景而不是阅读理解场景。
    return [t.strip() for t in jieba.lcut_for_search(text) if t.strip() and t.strip() not in {"，", "。", "、", "the", "is", ":", " "}]


@dataclass
class BM25Index:
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # 每个chunk来自哪个文件名
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_len: float = 0.0
        self._n: int = 0

    def add_documents(self, chunks: list[str], source: str) -> None:
        for c in chunks:
            self.chunks.append(c)
            self.sources.append(source)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._doc_tokens = [_tokenize(c) for c in self.chunks]
        self._n = len(self._doc_tokens)
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        total_len = sum(len(t) for t in self._doc_tokens)
        self._avg_doc_len = (total_len / self._n) if self._n else 0.0

    def _idf(self, term: str) -> float:
        n_qi = self._doc_freq.get(term, 0)
        return math.log((self._n - n_qi + 0.5) / (n_qi + 0.5) + 1)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self._n == 0:
            return []
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scores = [0.0] * self._n
        for idx, tokens in enumerate(self._doc_tokens):
            doc_len = len(tokens)
            term_freq = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in term_freq:
                    continue
                f = term_freq[term]
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len or 1))
                score += idf * (f * (self.k1 + 1)) / (denom or 1)
            scores[idx] = score

        ranked = sorted(range(self._n), key=lambda i: scores[i], reverse=True)
        results = []
        top_score = scores[ranked[0]] if ranked else 0
        # 弱相关过滤:得分低于最高分15%的结果大概率只是偶然词面重合,不视为有效命中,
        # 避免把不相关内容也当作"检索到的依据"返回给用户。
        relevance_floor = max(top_score * 0.15, 0.01)
        for i in ranked[:top_k]:
            if scores[i] < relevance_floor:
                continue
            results.append({
                "chunk_id": i,
                "score": round(scores[i], 4),
                "text": self.chunks[i],
                "source": self.sources[i],
            })
        return results
