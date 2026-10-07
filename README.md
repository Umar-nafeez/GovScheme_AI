# 🏛️ GovScheme AI

## Government Scheme Assistant powered by Hybrid RAG, Reranking, CRAG & Agentic Retrieval

GovScheme AI is an evidence-grounded AI assistant that helps users understand Indian government schemes by retrieving information from a curated collection of official government documents.

Unlike a basic RAG chatbot that performs a single vector search and directly sends the results to an LLM, GovScheme AI uses a multi-stage retrieval pipeline designed to improve retrieval quality and reduce unsupported answers.

The system combines:

- 🔎 Dense Vector Search
- 🔤 BM25 Keyword Search
- 🔀 Reciprocal Rank Fusion (RRF)
- 🎯 Cross-Encoder Reranking
- 🔄 Corrective RAG (CRAG)
- ✍️ Query Rewriting
- 🤖 Agentic Retrieval with LangGraph
- 🧠 Conversation Memory
- 📚 Grounded Answer Generation
- 🔗 Source and Page Citations

---

# 📌 Project Overview

Government schemes often have complex eligibility rules, financial limits, subsidies, application conditions, and implementation guidelines spread across multiple government documents.

A user may ask something simple like:

> "What government scheme can help me start an MSME?"

The challenge is not simply generating an answer.

The system first needs to:

1. Understand the user's question.
2. Find relevant government documents.
3. Retrieve information using both semantic and keyword-based methods.
4. Combine the retrieval results.
5. Rerank the retrieved documents.
6. Check whether the retrieved evidence is actually useful.
7. Rewrite the query if retrieval quality is poor.
8. Retrieve again.
9. Generate an answer using the retrieved evidence.
10. Show the sources used for the answer.

GovScheme AI was designed around this complete retrieval workflow.

---

# 🎯 Problem Statement

A basic RAG pipeline generally looks like:

```text
User Question
      ↓
Vector Search
      ↓
Top Documents
      ↓
LLM
      ↓
Answer