| Retrieval config | Recall@5 | Recall@5 (table queries) | ms/query | misses |
|---|---|---|---|---|
| bm25 only | 0.853 | 0.667 | 41 | r03, r16, r17, r19, r30 |
| vectors only | 0.706 | 0.333 | 359 | r01, r03, r14, r16, r19, r23, r26, r28, r30, r34 |
| hybrid (RRF) | 0.794 | 0.333 | 114 | r03, r14, r16, r19, r23, r30, r34 |
| hybrid + rerank | 0.882 | 1.0 | 665 | r16, r17, r19, r30 |
| hybrid + rerank + decomposition | 0.882 | 1.0 | 569 | r16, r17, r19, r30 |
| plain chunks: hybrid + rerank + decomposition | 0.853 | 0.667 | 635 | r03, r16, r17, r19, r30 |

_Multi-query (LLM rewrite) not measured yet: the OpenRouter key on the build machine returned 401; run `python -m agent_desk eval retrieval` with a valid key._
