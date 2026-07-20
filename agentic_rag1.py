# Claude approach.

"""
Iterative Agentic RAG with CAMEL-AI
------------------------------------
Pattern: Supervisor (deterministic Python loop) coordinates two agents:
  1. Retriever agent   - decides *what* to search the vector DB for, calls the search tool
  2. Reviewer agent     - judges whether accumulated context is sufficient to answer;
                          if not, proposes a follow-up query
The supervisor loops retriever -> reviewer until the reviewer says "sufficient"
or a max-iteration cap is hit, then calls a final answer agent.

Why not use CAMEL's Workforce for the whole thing?
Workforce is designed for open-ended task decomposition where an LLM coordinator
decides how to split work. Here we want a *deterministic* stopping condition
(max iterations, explicit sufficiency check), so a plain Python loop driving two
ChatAgents is more reliable and easier to debug than letting an LLM decide when
to stop looping. See the bottom of this file for a note on the Workforce alternative.
"""

import json
from typing import List, Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits import FunctionTool

# ---------------------------------------------------------------------------
# 1. Vector DB access (swap this for your actual client - Chroma, Azure AI
#    Search, pgvector, etc. Shown here as a thin wrapper so it can be
#    exposed to the retriever agent as a callable tool.)
# ---------------------------------------------------------------------------

class VectorDB:
    """Thin wrapper around your real vector store."""

    def __init__(self, collection):
        self.collection = collection  # e.g. a chromadb Collection

    def search(self, query: str, top_k: int = 3, exclude_ids: Optional[List[str]] = None) -> List[dict]:
        """
        Search the vector DB for chunks relevant to `query`.

        Args:
            query: natural-language search query
            top_k: number of chunks to return
            exclude_ids: chunk ids already retrieved in this session, so the
                         retriever doesn't fetch the same chunk twice on a
                         follow-up query
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k + (len(exclude_ids) if exclude_ids else 0),
        )
        chunks = []
        for doc_id, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        ):
            if exclude_ids and doc_id in exclude_ids:
                continue
            chunks.append({"id": doc_id, "text": doc, "metadata": meta})
            if len(chunks) >= top_k:
                break
        return chunks


# ---------------------------------------------------------------------------
# 2. Agent factory
# ---------------------------------------------------------------------------

def make_model():
    # Point this at whatever backend you're using - shown here as an
    # OpenAI-compatible endpoint; swap ModelPlatformType/ModelType for
    # Anthropic/Azure as needed.
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )


def build_retriever_agent(vector_db: VectorDB, retrieved_log: List[dict]) -> ChatAgent:
    def search_vector_db(query: str, top_k: int = 3) -> str:
        """Search the internal knowledge base for chunks relevant to a query."""
        exclude_ids = [c["id"] for c in retrieved_log]
        chunks = vector_db.search(query, top_k=top_k, exclude_ids=exclude_ids)
        retrieved_log.extend(chunks)
        return json.dumps(chunks, ensure_ascii=False)

    tool = FunctionTool(search_vector_db)

    system_msg = BaseMessage.make_assistant_message(
        role_name="Retriever",
        content=(
            "You retrieve information from a vector database using the "
            "search_vector_db tool. Given a query or a follow-up gap "
            "description, call the tool with a well-formed search query. "
            "Do not answer the user directly - only retrieve."
        ),
    )
    return ChatAgent(system_message=system_msg, model=make_model(), tools=[tool])


def build_reviewer_agent() -> ChatAgent:
    system_msg = BaseMessage.make_assistant_message(
        role_name="Reviewer",
        content=(
            "You judge whether the accumulated retrieved context is enough "
            "to fully answer the user's question. Respond ONLY with JSON in "
            "this exact schema, no prose outside the JSON:\n"
            '{"sufficient": true|false, "missing_info": "<what is still '
            'missing, empty string if sufficient>", "next_query": "<a '
            'concrete follow-up search query, empty string if sufficient>"}'
        ),
    )
    return ChatAgent(system_message=system_msg, model=make_model())


def build_answer_agent() -> ChatAgent:
    system_msg = BaseMessage.make_assistant_message(
        role_name="Answerer",
        content=(
            "You answer the user's question using only the provided context "
            "chunks. If context is genuinely insufficient after all retrieval "
            "attempts, say so explicitly rather than guessing."
        ),
    )
    return ChatAgent(system_message=system_msg, model=make_model())


# ---------------------------------------------------------------------------
# 3. Supervisor: deterministic control loop
# ---------------------------------------------------------------------------

def run_agentic_rag(user_question: str, vector_db: VectorDB, max_iterations: int = 3) -> str:
    retrieved_log: List[dict] = []
    retriever = build_retriever_agent(vector_db, retrieved_log)
    reviewer = build_reviewer_agent()
    answerer = build_answer_agent()

    query = user_question

    for iteration in range(1, max_iterations + 1):
        # --- Retriever step ---
        retriever_response = retriever.step(
            BaseMessage.make_user_message(role_name="User", content=query)
        )
        # retrieved_log has been appended to by the search_vector_db tool call

        # --- Reviewer step ---
        context_text = "\n\n".join(f"[{c['id']}] {c['text']}" for c in retrieved_log)
        review_prompt = (
            f"Original question: {user_question}\n\n"
            f"Accumulated context so far:\n{context_text}\n\n"
            "Is this sufficient to answer the question?"
        )
        review_response = reviewer.step(
            BaseMessage.make_user_message(role_name="User", content=review_prompt)
        )
        reviewer.reset()  # keep reviewer stateless between iterations

        try:
            verdict = json.loads(review_response.msgs[0].content)
        except (json.JSONDecodeError, IndexError):
            # If the reviewer's output isn't parseable, fail safe: stop looping
            break

        if verdict.get("sufficient"):
            break

        next_query = verdict.get("next_query") or ""
        if not next_query:
            break

        query = f"Follow-up needed: {verdict.get('missing_info', '')}. Search for: {next_query}"

    # --- Final answer ---
    final_context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in retrieved_log)
    answer_prompt = (
        f"Question: {user_question}\n\nContext:\n{final_context}\n\nAnswer the question."
    )
    answer_response = answerer.step(
        BaseMessage.make_user_message(role_name="User", content=answer_prompt)
    )
    return answer_response.msgs[0].content


# ---------------------------------------------------------------------------
# 4. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("my_docs")
    # ... populate `collection` with your documents beforehand ...

    vdb = VectorDB(collection)
    answer = run_agentic_rag(
        "What was the approval process change introduced in the 2024 policy update, "
        "and which department is now responsible for sign-off?",
        vdb,
        max_iterations=3,
    )
    print(answer)


# ---------------------------------------------------------------------------
# Alternative: CAMEL Workforce version (more autonomous, less deterministic)
# ---------------------------------------------------------------------------
#
# from camel.societies.workforce import Workforce
#
# workforce = Workforce("Agentic RAG Team")
# workforce.add_single_agent_worker(
#     "Retriever who searches the vector DB for relevant chunks",
#     worker=build_retriever_agent(vector_db, retrieved_log),
# ).add_single_agent_worker(
#     "Reviewer who judges if retrieved context is sufficient and requests "
#     "follow-up retrieval if not",
#     worker=build_reviewer_agent(),
# )
# result = workforce.process_task(user_question)
#
# This lets the Workforce coordinator LLM decide the routing/looping itself,
# which is more flexible but harder to bound (no guaranteed max_iterations,
# and failure modes are "the coordinator decided wrong" rather than a bug you
# can step through). Recommended only once you've validated the deterministic
# version and want more open-ended behavior.