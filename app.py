import os
import uuid
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GovScheme AI",
    page_icon="🏛️",
    layout="centered"
)


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Check your .env file.")
    st.stop()


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }

    .source-box {
        padding: 0.7rem;
        margin-top: 0.5rem;
        border-left: 3px solid #6366f1;
        background: rgba(99,102,241,0.07);
        border-radius: 6px;
        font-size: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vector_db():

    embeddings = load_embeddings()

    return Chroma(
        persist_directory=str(
            PROJECT_ROOT / "data" / "chroma_db"
        ),
        embedding_function=embeddings
    )


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@st.cache_resource
def load_llm():

    return ChatGroq(
        model="qwen/qwen3.6-27b",
        temperature=0,
        max_tokens=800,
        reasoning_effort="none",
        api_key=GROQ_API_KEY
    )


embeddings = load_embeddings()
vector_db = load_vector_db()
reranker = load_reranker()
llm = load_llm()


# ============================================================
# LOAD DOCUMENTS FOR BM25
# ============================================================

@st.cache_resource
def load_documents():

    data = vector_db.get(
        include=["documents", "metadatas"]
    )

    documents = []

    for text, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    documents.sort(
        key=lambda doc: int(doc.metadata["chunk_id"])
    )

    return documents


chunks = load_documents()


# ============================================================
# BM25
# ============================================================

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query, k=10):

    # Dense retrieval
    dense_results = vector_db.similarity_search(
        query,
        k=k
    )

    dense_ids = [
        int(doc.metadata["chunk_id"])
        for doc in dense_results
    ]

    # BM25 retrieval
    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=bm25_scores.__getitem__,
        reverse=True
    )[:k]

    bm25_ids = [
        int(chunks[index].metadata["chunk_id"])
        for index in bm25_indices
    ]

    # RRF
    rrf_scores = {}

    for results in [dense_ids, bm25_ids]:

        for rank, chunk_id in enumerate(
            results,
            start=1
        ):

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0)
                + 1 / (60 + rank)
            )

    # Final hybrid ranking
    hybrid_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:k]

    chunk_map = {
        int(chunk.metadata["chunk_id"]): chunk
        for chunk in chunks
    }

    return [
        chunk_map[chunk_id]
        for chunk_id in hybrid_ids
    ]


# ============================================================
# RERANKING
# ============================================================

def rerank_documents(
    query,
    documents,
    k=3
):

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        document
        for document, score in ranked_documents[:k]
    ]


# ============================================================
# CRAG
# ============================================================

class RetrievalDecision(BaseModel):

    decision: Literal[
        "correct",
        "ambiguous",
        "incorrect"
    ]

    reason: str


evaluator = llm.with_structured_output(
    RetrievalDecision
)


def evaluate_retrieval(
    query,
    documents
):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a retrieval evaluator for a government scheme assistant.

User question:
{query}

Retrieved documents:
{context}

Determine whether the retrieved documents provide useful evidence
to answer the user's question.

correct = sufficient and relevant evidence
ambiguous = partially relevant or insufficient evidence
incorrect = irrelevant evidence

Return the decision and a short reason.
"""

    return evaluator.invoke(prompt)


def rewrite_query(
    query,
    reason
):

    prompt = f"""
Rewrite the following user query to improve document retrieval.

Original query:
{query}

Reason retrieval was not good enough:
{reason}

Create a clearer search query while preserving the user's intent.

Return only the rewritten query.
"""

    return llm.invoke(prompt).content.strip()


# ============================================================
# LANGGRAPH — AGENTIC RETRIEVAL + MEMORY
# ============================================================

class RetrievalState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]

    query: str
    documents: list
    decision: str
    reason: str
    retry_count: int


def retrieve_documents(
    state: RetrievalState
):

    documents = hybrid_search(
        state["query"],
        k=10
    )

    documents = rerank_documents(
        state["query"],
        documents,
        k=3
    )

    return {
        "documents": documents
    }


def evaluate_documents(
    state: RetrievalState
):

    result = evaluate_retrieval(
        state["query"],
        state["documents"]
    )

    return {
        "decision": result.decision,
        "reason": result.reason
    }


def rewrite_query_node(
    state: RetrievalState
):

    new_query = rewrite_query(
        state["query"],
        state["reason"]
    )

    return {
        "query": new_query,
        "retry_count": state["retry_count"] + 1
    }


def retrieval_decision(
    state: RetrievalState
):

    if state["decision"] == "correct":
        return "done"

    if state["retry_count"] >= 1:
        return "done"

    return "rewrite"


graph = StateGraph(
    RetrievalState
)

graph.add_node(
    "retrieve",
    retrieve_documents
)

graph.add_node(
    "evaluate",
    evaluate_documents
)

graph.add_node(
    "rewrite",
    rewrite_query_node
)

graph.add_edge(
    START,
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "evaluate"
)

graph.add_conditional_edges(
    "evaluate",
    retrieval_decision,
    {
        "done": END,
        "rewrite": "rewrite"
    }
)

graph.add_edge(
    "rewrite",
    "retrieve"
)


memory = InMemorySaver()

agentic_retriever = graph.compile(
    checkpointer=memory
)


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    query,
    documents
):

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    prompt = f"""
You are GovScheme AI, an assistant that helps users understand
Indian government schemes.

Answer the user's question using ONLY the provided government
documents.

Do not use outside knowledge.

If the documents do not contain enough information, say:

"I don't have enough information in the provided documents
to answer that confidently."

Answer clearly and naturally.

User question:
{query}

Government documents:
{context}
"""

    return llm.invoke(prompt).content


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def ask_govscheme(
    query,
    thread_id
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Get previous conversation from LangGraph memory
    state = agentic_retriever.get_state(
        config
    )

    previous_messages = state.values.get(
        "messages",
        []
    )

    # Convert follow-up question into
    # a standalone retrieval query
    if previous_messages:

        history = "\n".join(
            f"{message.type}: {message.content}"
            for message in previous_messages[-6:]
        )

        rewrite_prompt = f"""
Convert the user's latest question into a
standalone search query using the conversation context.

Conversation:
{history}

Latest question:
{query}

Return only the standalone search query.
"""

        search_query = llm.invoke(
            rewrite_prompt
        ).content.strip()

    else:

        search_query = query


    # Agentic retrieval
    result = agentic_retriever.invoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ],
            "query": search_query,
            "retry_count": 0
        },
        config=config
    )


    # Generate final answer using
    # the ORIGINAL user question
    answer = generate_answer(
        query,
        result["documents"]
    )


    # Store assistant response in
    # LangGraph conversation memory
    agentic_retriever.update_state(
        config,
        {
            "messages": [
                AIMessage(
                    content=answer
                )
            ]
        }
    )


    return answer, result["documents"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏛️ GovScheme AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Government Scheme Assistant powered by Hybrid RAG,
    Reranking, CRAG and Agentic Retrieval
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">
                        <b>{source["source"]}</b><br>
                        Page: {source["page"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask about government schemes..."
)


if query:

    # Display user message
    with st.chat_message("user"):

        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching government schemes..."
        ):

            answer, documents = ask_govscheme(
                query,
                st.session_state.thread_id
            )

        st.markdown(answer)


        # Sources
        sources = []

        for document in documents:

            sources.append(
                {
                    "source": document.metadata.get(
                        "source",
                        "Unknown"
                    ),
                    "page": document.metadata.get(
                        "page",
                        "Unknown"
                    )
                }
            )

        sources = list(
            {
                (
                    source["source"],
                    source["page"]
                ): source
                for source in sources
            }.values()
        )


        if sources:

            with st.expander(
                "Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"""
                        <div class="source-box">
                        <b>{source["source"]}</b><br>
                        Page: {source["page"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )