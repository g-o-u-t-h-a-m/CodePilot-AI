GitHub URL

↓

Repository Manager

↓

Repository Scanner

↓

SourceFile

↓

Chunk Engine

↓

CodeChunk

↓

Embedding Engine

↓

Vector Store

↓

Retriever

↓

Prompt Builder

↓

LLM

↓

Answer

## Sprint 8: Generation Layer

The Retrieval-Augmented Generation pipeline is now complete end-to-end:

```
User Question
    ↓
Retriever                       (RetrievalResult[], Sprint 7)
    ↓
PromptBuilder                   (grounded, budgeted prompt, Sprint 8)
    ↓
LLMProvider                     (pluggable; mock or OpenAI-compatible, Sprint 8)
    ↓
Grounded Answer                 (RagResponse, Sprint 8)
```

Key Sprint 8 components:

- **PromptBuilder** (`app/prompts/`) — converts a question, repository name,
  and retrieved chunks into a deterministic, inspectable prompt that
  instructs the model to ground every claim in the supplied context
  (grounding rules, file paths, line ranges) and to state when evidence is
  insufficient. It enforces a configurable context budget and only ever
  includes complete chunks.
- **LLMProvider** (`app/llm/`) — provider abstraction (Strategy + Registry
  patterns, mirroring the embeddings architecture). `MockLLMProvider`
  (default) enables full local, no-cost testing; `OpenAICompatibleProvider`
  is an optional, configurable HTTP bridge to OpenAI-compatible services.
- **RAGService** (`app/rag/service.py`) — orchestrates
  validate → retrieve → build prompt → generate → structured answer,
  depending only on the Retriever, PromptBuilder, and LLMProvider
  abstractions. It applies a relevance gate so questions with no genuinely
  relevant retrieved context return an explicit insufficient-context
  response rather than fabricating code.