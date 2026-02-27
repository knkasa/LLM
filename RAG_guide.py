# RAG ultimate guide.
# https://pub.towardsai.net/i-spent-3-months-building-ra-systems-before-learning-these-11-strategies-1a8f6b4278aa
#RRF=use this to rerank hybrid RAG search.

#Strategy 1: Context-Aware Chunking
#What it does: Instead of splitting documents at fixed character counts, it analyzes semantic boundaries and document structure.
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

class SmartChunker:
    def __init__(self, max_tokens=512):
        # Use actual tokenizer, not character counts
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            merge_peers=True  # Combine small adjacent chunks
        )
  
    def chunk_document(self, document):
        # Analyzes document structure (headings, paragraphs, tables)
        chunks = list(self.chunker.chunk(dl_doc=document))
  
        # Each chunk includes heading context
        contextualized_chunks = []
        for chunk in chunks:
            # Adds hierarchical heading information
            contextualized_text = self.chunker.contextualize(chunk=chunk)
            contextualized_chunks.append(contextualized_text)
  
        return contextualized_chunks


#Strategy 2: Contextual Retrieval
#What it does: Adds document-level context to each chunk before embedding. An LLM generates 1–2 sentences explaining what each chunk discusses in relation to the whole document.
async def enrich_chunk(chunk: str, document: str, title: str) -> str:
    """Add contextual prefix using LLM"""
    prompt = f"""
Title: {title}
{document[:4000]}
{chunk}

Provide brief context (1-2 sentences) explaining what this chunk discusses  
in relation to the full document. Format: "This chunk from [title] discusses [explanation]." """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )
  
    context = response.choices[0].message.content.strip()
  
    # Embed the contextualized version
    return f"{context}\n\n{chunk}"

#Strategy 3: Re-ranking
#What it does: Two-stage retrieval where fast vector search finds 20–50 candidates, then a cross-encoder model re-scores them for precision.
from sentence_transformers import CrossEncoder

# Initialize once
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
async def search_with_reranking(query: str, limit: int = 5) -> list:
    # Stage 1: Fast vector retrieval (get 4x candidates)
    candidate_limit = min(limit * 4, 20)
    query_embedding = await embedder.embed_query(query)
  
    candidates = await db.query(
        "SELECT content, metadata FROM chunks ORDER BY embedding  $1 LIMIT $2",
        query_embedding, candidate_limit
    )
  
    # Stage 2: Re-rank with cross-encoder
    pairs = [[query, row['content']] for row in candidates]
    scores = reranker.predict(pairs)
  
    # Sort by reranker scores and return top N
    reranked = sorted(
        zip(candidates, scores),  
        key=lambda x: x[1],  
        reverse=True
    )[:limit]
  
    return [doc for doc, score in reranked]

#Strategy 4: Query Expansion
#What it does: Expands a brief query into a more detailed, comprehensive version using an LLM.
async def expand_query(query: str) -> str:
    """Expand brief query into detailed version"""
    system_prompt = """You are a query expansion assistant.  
Take brief user queries and expand them into more detailed versions that:
1. Add relevant context and clarifications
2. Include related terminology and concepts
3. Specify what aspects should be covered
4. Maintain the original intent
5. Keep it as a single, coherent question
Expand the query to be 2-3x more detailed while staying focused."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Expand this query: {query}"}
        ],
        temperature=0.3
    )
  
    return response.choices[0].message.content.strip()


#Strategy 5: Multi-Query RAG
#What it does: Generates 3–4 different phrasings of the same question, searches with all of them in parallel, and deduplicates results.
async def search_with_multi_query(query: str, limit: int = 5) -> list:
    # Generate query variations
    variations_prompt = f"""Generate 3 different phrasings of this query:
    "{query}"
  
    Return only the 3 queries, one per line."""
  
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": variations_prompt}],
        temperature=0.7
    )
  
    queries = [query] + response.choices[0].message.content.strip().split('\n')
  
    # Execute all searches in parallel
    search_tasks = []
    for q in queries:
        query_embedding = await embedder.embed_query(q)
        task = db.fetch(
            "SELECT * FROM match_chunks($1::vector, $2)",  
            query_embedding, limit
        )
        search_tasks.append(task)
  
    results_lists = await asyncio.gather(*search_tasks)
  
    # Deduplicate by chunk ID, keeping highest similarity
    seen = {}
    for results in results_lists:
        for row in results:
            chunk_id = row['chunk_id']
            if chunk_id not in seen or row['similarity'] > seen[chunk_id]['similarity']:
                seen[chunk_id] = row
  
    # Return top N unique results
    return sorted(
        seen.values(),  
        key=lambda x: x['similarity'],  
        reverse=True
    )[:limit]

#Strategy 6: Agentic RAG
#What it does: Gives the AI agent multiple retrieval tools and lets it autonomously choose which to use based on the query.
from pydantic_ai import Agent
agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a RAG assistant with multiple retrieval tools. Choose the right tool(s) for each query.'
)
@agent.tool
async def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Semantic search over document chunks"""
    query_embedding = await embedder.embed_query(query)
    results = await db.match_chunks(query_embedding, limit)
    return format_results(results)
@agent.tool
async def retrieve_full_document(document_title: str) -> str:
    """Retrieve complete document when chunks lack context"""
    result = await db.query(
        "SELECT title, content FROM documents WHERE title ILIKE %s",
        f"%{document_title}%"
    )
    return f"**{result['title']}**\n\n{result['content']}"
@agent.tool
async def sql_query(question: str) -> str:
    """Query structured database for specific data"""
    # Agent can write SQL queries for structured data
    # (In production, use proper SQL generation with safety checks)
    return execute_safe_sql(question)

#Strategy 7: Self-Reflective RAG
#What it does: After retrieving documents, the system evaluates their relevance, refines the query if needed, and re-searches until satisfied.
async def search_with_self_reflection(query: str, limit: int = 5, max_iterations: int = 2) -> dict:
    """Self-correcting search loop"""
  
    for iteration in range(max_iterations):
        # Perform search
        results = await vector_search(query, limit)
  
        # Grade relevance
        grade_prompt = f"""Query: {query}
  
Retrieved documents:
{format_docs_for_grading(results)}
Grade the relevance of these documents to the query on a scale of 1-5.
Respond with only the number."""
        grade_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": grade_prompt}],
            temperature=0
        )
  
        grade = int(grade_response.choices[0].message.content.strip().split()[0])
  
        # If good results, return them
        if grade >= 3:
            return {
                "results": results,
                "iterations": iteration + 1,
                "final_query": query
            }
  
        # If poor results and not last iteration, refine query
        if iteration             refine_prompt = f"""Query "{query}" returned low-relevance results.
  
Suggest an improved query that might find better documents.
Respond with only the improved query."""
            refined_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.5
            )
  
            query = refined_response.choices[0].message.content.strip()
  
    # Return best attempt
    return {
        "results": results,
        "iterations": max_iterations,
        "final_query": query
    }

#Strategy 8: Knowledge Graphs
#What it does: Combines vector search with graph databases to capture relationships between entities.
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Initialize Graphiti (connects to Neo4j)
graphiti = Graphiti("neo4j://localhost:7687", "neo4j", "password")
async def ingest_document(text: str, source: str):
    """Ingest into knowledge graph"""
    # Graphiti automatically extracts entities and relationships
    await graphiti.add_episode(
        name=source,
        episode_body=text,
        source=EpisodeType.text,
        source_description=f"Document: {source}"
    )
async def search_knowledge_graph(query: str) -> str:
    """Hybrid search: semantic + keyword + graph"""
    # Graphiti combines:
    # - Semantic similarity (embeddings)
    # - BM25 keyword search
    # - Graph structure traversal
    # - Temporal context
  
    results = await graphiti.search(query=query, num_results=5)
  
    # Format graph results
    formatted = []
    for result in results:
        formatted.append(
            f"Entity: {result.node.name}\n"
            f"Type: {result.node.type}\n"
            f"Relationships: {result.relationships}"
        )
  
    return "\n---\n".join(formatted)

#Strategy 9: Hierarchical RAG
#What it does: Creates parent-child chunk relationships. Search small chunks for precision, return large parent chunks for context.
def ingest_hierarchical(document: str, title: str):
    """Create parent-child structure"""
    # Parents: large sections (2000 chars)
    parent_chunks = [document[i:i+2000] for i in range(0, len(document), 2000)]
  
    for parent_id, parent in enumerate(parent_chunks):
        # Store parent
        metadata = {"heading": f"{title} - Section {parent_id}"}
        db.execute(
            "INSERT INTO parent_chunks (id, content, metadata) VALUES (%s, %s, %s)",
            (parent_id, parent, json.dumps(metadata))
        )
  
        # Children: small chunks (500 chars)
        child_chunks = [parent[j:j+500] for j in range(0, len(parent), 500)]
        for child in child_chunks:
            embedding = get_embedding(child)
            db.execute(
                "INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)",
                (child, embedding, parent_id)
            )

async def hierarchical_search(query: str) -> str:
    """Search children, return parents"""
    query_emb = get_embedding(query)
  
    # Search small children for precision
    results = await db.query(
        """SELECT p.content, p.metadata
           FROM child_chunks c
           JOIN parent_chunks p ON c.parent_id = p.id
           ORDER BY c.embedding  %s LIMIT 3""",
        query_emb
    )
  
    # Return large parents for context
    formatted = []
    for content, metadata in results:
        meta = json.loads(metadata)
        formatted.append(f"[{meta['heading']}]\n{content}")
  
    return "\n\n".join(formatted)

#Strategy 10: Late Chunking
#What it does: Processes the entire document through the transformer before chunking the token embeddings (not the text).
def late_chunk(text: str, chunk_size=512) -> list:
    """Embed full document BEFORE chunking"""
  
    # Step 1: Embed entire document (8192 tokens max)
    full_doc_token_embeddings = transformer_embed(text)  # Token-level
  
    # Step 2: Define chunk boundaries
    tokens = tokenize(text)
    chunk_boundaries = range(0, len(tokens), chunk_size)
  
    # Step 3: Pool token embeddings for each chunk
    chunks_with_embeddings = []
    for start in chunk_boundaries:
        end = start + chunk_size
        chunk_text = detokenize(tokens[start:end])
  
        # Mean pool token embeddings (preserves full doc context!)
        chunk_embedding = mean_pool(full_doc_token_embeddings[start:end])
        chunks_with_embeddings.append((chunk_text, chunk_embedding))
  
    return chunks_with_embeddings

#Strategy 11: Fine-tuned Embeddings
#What it does: Trains embedding models on domain-specific query-document pairs.
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

def prepare_training_data():
    """Domain-specific query-document pairs"""
    return [
        ("What is EBITDA?", "EBITDA (Earnings Before Interest, Taxes..."),
        ("Explain capital expenditure", "Capital expenditure (CapEx) refers to..."),
        # ... thousands more pairs
    ]
def fine_tune_model():
    """Fine-tune on domain data"""
    # Load base model
    model = SentenceTransformer('all-MiniLM-L6-v2')
  
    # Prepare training data
    train_examples = prepare_training_data()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
  
    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)
  
    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100
    )
  
    model.save('./fine_tuned_financial_model')
    return model
# Use fine-tuned model
embedding_model = SentenceTransformer('./fine_tuned_financial_model')


#Combo 1: The Production-Ready Stack (Best Overall)
#Strategies: Context-Aware Chunking + Re-ranking + Query Expansion + Agentic RAG
from pydantic_ai import Agent
from sentence_transformers import CrossEncoder

class ProductionRAG:
    def __init__(self):
        self.agent = Agent('openai:gpt-4o')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  
    async def ingest(self, document: str, title: str):
        """Context-aware chunking"""
        chunker = DoclingHybridChunker(max_tokens=512)
        chunks = chunker.chunk_document(document)
  
        for chunk in chunks:
            embedding = await self.embed(chunk)
            await self.db.insert(chunk, embedding, title)
  
    @agent.tool
    async def search_knowledge_base(self, query: str, limit: int = 5) -> str:
        """Query expansion + Re-ranking"""
        # Step 1: Expand query
        expanded_query = await self.expand_query(query)
  
        # Step 2: Get candidates (4x more than needed)
        query_embedding = await self.embed(expanded_query)
        candidates = await self.db.search(query_embedding, limit * 4)
  
        # Step 3: Re-rank
        pairs = [[expanded_query, doc['content']] for doc in candidates]
        scores = self.reranker.predict(pairs)
  
        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
  
        return self.format_results(reranked)
  
    @agent.tool
    async def retrieve_full_document(self, title: str) -> str:
        """Agentic: retrieve complete documents when needed"""
        doc = await self.db.get_full_document(title)
        return f"**{doc['title']}**\n\n{doc['content']}"
  
    async def query(self, user_query: str) -> str:
        """Agent chooses best retrieval strategy"""
        result = await self.agent.run(user_query)
        return result.data

rag = ProductionRAG()
answer = await rag.query("What were the key factors in Q2 revenue growth?")

#Combo 2: The High-Accuracy Stack (Best for Critical Applications)
#Strategies: Contextual Retrieval + Multi-Query + Re-ranking + Self-Reflective RAG
class HighAccuracyRAG:
    def __init__(self):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  
    async def ingest(self, document: str, title: str):
        """Contextual retrieval during ingestion"""
        chunks = self.smart_chunk(document)
  
        for chunk in chunks:
            # Add document context
            enriched = await self.enrich_chunk(chunk, document, title)
            embedding = await self.embed(enriched)
            await self.db.insert(enriched, embedding, title)
  
    async def search(self, query: str, limit: int = 5) -> dict:
        """Multi-query + Re-ranking + Self-reflection"""
        # Step 1: Generate query variations
        queries = await self.generate_query_variations(query)
  
        # Step 2: Parallel search with all variations
        all_candidates = []
        for q in queries:
            embedding = await self.embed(q)
            results = await self.db.search(embedding, limit * 4)
            all_candidates.extend(results)
  
        # Deduplicate
        unique_candidates = self.deduplicate(all_candidates)
  
        # Step 3: Re-rank all candidates
        pairs = [[query, doc['content']] for doc in unique_candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(unique_candidates, scores), key=lambda x: x[1], reverse=True)
  
        # Step 4: Self-reflection loop
        for iteration in range(2):
            top_results = reranked[:limit]
  
            # Grade relevance
            grade = await self.grade_results(query, top_results)
  
            if grade >= 4:  # Good results
                return {
                    "results": top_results,
                    "iterations": iteration + 1,
                    "confidence": "high"
                }
  
            # Refine and retry if needed
            if iteration                 refined_query = await self.refine_query(query, top_results)
                query = refined_query
                # ... repeat search with refined query
  
        return {"results": reranked[:limit], "iterations": 2, "confidence": "medium"}

rag = HighAccuracyRAG()
result = await rag.search("What are the contraindications for this medication?")

#Combo 3: The Domain Expert Stack (Best for Specialized Fields)
#Strategies: Fine-tuned Embeddings + Contextual Retrieval + Knowledge Graphs + Re-ranking
from graphiti_core import Graphiti
class DomainExpertRAG:
    def __init__(self, domain_model_path: str):
        # Fine-tuned embeddings
        self.embedder = SentenceTransformer(domain_model_path)
  
        # Knowledge graph
        self.graph = Graphiti("neo4j://localhost:7687", "neo4j", "password")
  
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  
    async def ingest(self, document: str, title: str):
        """Contextual retrieval + Knowledge graph"""
        chunks = self.smart_chunk(document)
  
        for chunk in chunks:
            # Add domain context
            enriched = await self.enrich_chunk(chunk, document, title)
  
            # Fine-tuned embeddings
            embedding = self.embedder.encode(enriched)
            await self.db.insert(enriched, embedding, title)
  
        # Build knowledge graph
        await self.graph.add_episode(
            name=title,
            episode_body=document,
            source=EpisodeType.text
        )
  
    async def search(self, query: str, limit: int = 5) -> dict:
        """Hybrid: Vector (fine-tuned) + Graph + Re-ranking"""
        # Step 1: Fine-tuned embedding search
        query_embedding = self.embedder.encode(query)
        vector_results = await self.db.search(query_embedding, limit * 2)
  
        # Step 2: Knowledge graph search
        graph_results = await self.graph.search(query, num_results=limit * 2)
  
        # Step 3: Combine results
        all_candidates = self.merge_vector_and_graph(vector_results, graph_results)
  
        # Step 4: Re-rank with domain awareness
        pairs = [[query, doc['content']] for doc in all_candidates]
        scores = self.reranker.predict(pairs)
  
        reranked = sorted(
            zip(all_candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
  
        return {
            "results": reranked,
            "sources": ["vector", "graph"],
            "domain_model": domain_model_path
        }
# Usage (for medical domain)
rag = DomainExpertRAG(domain_model_path='./medical_embeddings')
result = await rag.search("Drug interactions between ACE inhibitors and NSAIDs")







