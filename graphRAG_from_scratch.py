# GraphRAG from scratch.
# https://medium.com/@DevBoostLab/graphrag-biggest-upgrade-ai-development-2026-33366891525d

'''
Graph RAG explained.

Query: "What did Dr. Sarah Smith work on?"

Query Entity: [Dr. Sarah Smith]
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
   [cardiac surgery]      [Hospital X]  ← Hop 1
         ↓                     ↓
   [surgical tools]      [Dr. Michael Smith]  ← Hop 2


## What Gets Fed to LLM
-------------------------------------------
Entities in query: Dr. Sarah Smith

Related entities (2-hop neighborhood):
- cardiac surgery (connected via: performed)
- Hospital X (connected via: works_at)
- Dr. Michael Smith (connected via: Hospital X employs)
- surgical tools (connected via: cardiac surgery requires)

Source chunks mentioning these entities:
[Chunk 1]: "Dr. Sarah Smith performed cardiac surgery..."
[Chunk 2]: "Hospital X employs several specialists..."

What did Dr. Sarah Smith work on?
--------------------------------------------
'''

#Note: Canonical entity is the word that gets cleaned from original word.
#eg. Dr. S. Smith -> Dr. Sarah Smith

#In the code below, it is using LLM to extract entities and nodes.
#Instead, you could use janome or spacy to extract entites.
#Then, finds subject/verb/nouns comibations again using janome/spacy.


from anthropic import Anthropic
import numpy as np
from sklearn.cluster import DBSCAN
import json

client = Anthropic(api_key="your-key")

#-------- Part 1 Entity extraction ---------------------------
class EntityResolver:
    """Production entity resolution with context-aware disambiguation"""
    
    def __init__(self):
        self.entity_cache = {}
        self.canonical_map = {}
        
    def extract_entities_with_context(self, text, chunk_id):
        """Extract entities with surrounding context for disambiguation"""
        
        prompt = f"""Extract ALL entities from this text. For each entity, provide:
1. Entity surface form (exact text)
2. Entity type (Person, Organization, Location, Product, Concept)
3. Surrounding context (the sentence containing the entity)
4. Disambiguation features (titles, roles, dates, locations mentioned nearby)

Text: {text}

Return JSON array:
[
  {{
    "surface_form": "Dr. Smith",
    "type": "Person",
    "context": "Dr. Smith performed the cardiac surgery on Tuesday",
    "features": {{"specialty": "cardiology", "title": "doctor"}}
  }}
]"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        entities = json.loads(response.content[0].text)
        
        # Store with context
        for entity in entities:
            entity['chunk_id'] = chunk_id
            
        return entities
    
    def compute_entity_similarity(self, entity1, entity2):
        """Compute similarity considering both text and semantic context"""
        
        # Exact match gets high score
        if entity1['surface_form'].lower() == entity2['surface_form'].lower():
            base_score = 0.9
        else:
            # Fuzzy match on surface form
            from difflib import SequenceMatcher
            base_score = SequenceMatcher(
                None, 
                entity1['surface_form'].lower(), 
                entity2['surface_form'].lower()
            ).ratio()
        
        # Type mismatch penalty
        if entity1['type'] != entity2['type']:
            base_score *= 0.3
        
        # Context similarity boost
        if 'features' in entity1 and 'features' in entity2:
            shared_features = set(entity1['features'].keys()) & set(entity2['features'].keys())
            if shared_features:
                # Features match increases confidence
                feature_match_score = sum(
                    1 for k in shared_features 
                    if entity1['features'][k] == entity2['features'][k]
                ) / len(shared_features)
                base_score = 0.7 * base_score + 0.3 * feature_match_score
        
        return base_score
    
    def resolve_entities(self, all_entities, similarity_threshold=0.75):
        """Cluster entities into canonical forms using DBSCAN"""
        
        n = len(all_entities)
        if n == 0:
            return {}
        
        # Build similarity matrix
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                sim = self.compute_entity_similarity(all_entities[i], all_entities[j])
                similarity_matrix[i,j] = sim
                similarity_matrix[j,i] = sim
        
        # Convert similarity to distance for DBSCAN
        distance_matrix = 1 - similarity_matrix
        
        # Cluster entities
        clustering = DBSCAN(
            eps=1-similarity_threshold, 
            min_samples=1, 
            metric='precomputed'
        ).fit(distance_matrix)
        
        # Create canonical entities
        canonical_entities = {}
        for cluster_id in set(clustering.labels_):
            cluster_members = [
                all_entities[i] for i, label in enumerate(clustering.labels_) 
                if label == cluster_id
            ]
            
            # Most common surface form becomes canonical
            surface_forms = [e['surface_form'] for e in cluster_members]
            canonical_form = max(set(surface_forms), key=surface_forms.count)
            
            canonical_entities[canonical_form] = {
                'canonical_name': canonical_form,
                'type': cluster_members[0]['type'],
                'variant_forms': list(set(surface_forms)),
                'occurrences': len(cluster_members),
                'contexts': [e['context'] for e in cluster_members[:5]]  # Sample contexts
            }
            
            # Map all variants to canonical form
            for variant in surface_forms:
                self.canonical_map[variant] = canonical_form
        
        return canonical_entities
    
    def get_canonical_form(self, surface_form):
        """Get canonical entity name for any surface form"""
        return self.canonical_map.get(surface_form, surface_form)

# Usage
resolver = EntityResolver()

documents = [
    ("chunk_1", "Dr. Sarah Smith performed the cardiac surgery."),
    ("chunk_2", "Dr. S. Smith, cardiologist, reviewed the results."),
    ("chunk_3", "Dr. Michael Smith, oncologist, recommended treatment."),
    ("chunk_4", "Oncologist Smith suggested chemotherapy.")
]

# Extract entities from all documents
all_entities = []
for chunk_id, text in documents:
    entities = resolver.extract_entities_with_context(text, chunk_id)
    all_entities.extend(entities)

# Resolve to canonical forms
canonical_entities = resolver.resolve_entities(all_entities)

print(f"Found {len(canonical_entities)} unique entities")
for canonical, data in canonical_entities.items():
    print(f"\nCanonical: {canonical}")
    print(f"  Variants: {data['variant_forms']}")
    print(f"  Type: {data['type']}")
    print(f"  Occurrences: {data['occurrences']}")


#-------- Part 2 Graph construction ------------------------------
import networkx as nx
from typing import List, Dict, Tuple

class GraphConstructor:
    """Build knowledge graph with resolved entities"""
    
    def __init__(self, entity_resolver):
        self.resolver = entity_resolver
        self.graph = nx.MultiDiGraph()
        self.entity_to_chunks = {}
        
    def extract_relationships(self, text, entities_in_chunk):
        """Extract relationships between resolved entities"""
        
        # Get canonical forms
        canonical_entities = [
            self.resolver.get_canonical_form(e['surface_form']) 
            for e in entities_in_chunk
        ]
        
        if len(canonical_entities) < 2:
            return []
        
        prompt = f"""Given these entities: {', '.join(canonical_entities)}

Analyze this text and extract relationships:
Text: {text}

Return JSON array of relationships:
[
  {{
    "source": "Entity A",
    "relation": "employed_by",
    "target": "Entity B",
    "evidence": "specific sentence showing relationship",
    "confidence": 0.95
  }}
]

Only extract relationships explicitly stated in the text."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        relationships = json.loads(response.content[0].text)
        
        # Canonicalize entity names in relationships
        for rel in relationships:
            rel['source'] = self.resolver.get_canonical_form(rel['source'])
            rel['target'] = self.resolver.get_canonical_form(rel['target'])
        
        return relationships
    
    def add_to_graph(self, chunk_id, chunk_text, entities, relationships):
        """Add entities and relationships to graph"""
        
        # Add entity nodes
        for entity in entities:
            canonical = self.resolver.get_canonical_form(entity['surface_form'])
            
            if canonical not in self.graph:
                self.graph.add_node(
                    canonical,
                    type=entity['type'],
                    contexts=[],
                    chunk_ids=[]
                )
            
            # Track which chunks mention this entity
            if canonical not in self.entity_to_chunks:
                self.entity_to_chunks[canonical] = []
            self.entity_to_chunks[canonical].append(chunk_id)
            
            # Add context
            self.graph.nodes[canonical]['contexts'].append(entity['context'])
            self.graph.nodes[canonical]['chunk_ids'].append(chunk_id)
        
        # Add relationship edges
        for rel in relationships:
            if rel['source'] in self.graph and rel['target'] in self.graph:
                self.graph.add_edge(
                    rel['source'],
                    rel['target'],
                    relation=rel['relation'],
                    evidence=rel['evidence'],
                    confidence=rel.get('confidence', 0.8),
                    chunk_id=chunk_id
                )
    
    def get_entity_neighborhood(self, entity_name, hops=2):
        """Get N-hop neighborhood for an entity"""
        
        canonical = self.resolver.get_canonical_form(entity_name)
        
        if canonical not in self.graph:
            return None
        
        # BFS to collect neighborhood
        visited = set()
        queue = [(canonical, 0)]
        neighborhood = {
            'nodes': [],
            'edges': [],
            'chunks': set()
        }
        
        while queue:
            node, depth = queue.pop(0)
            
            if node in visited or depth > hops:
                continue
            
            visited.add(node)
            
            # Add node data
            node_data = self.graph.nodes[node]
            neighborhood['nodes'].append({
                'name': node,
                'type': node_data['type'],
                'chunks': node_data.get('chunk_ids', [])
            })
            
            # Add edges
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                for key, attrs in edge_data.items():
                    neighborhood['edges'].append({
                        'source': node,
                        'target': neighbor,
                        'relation': attrs['relation'],
                        'evidence': attrs['evidence']
                    })
                    neighborhood['chunks'].add(attrs.get('chunk_id'))
                
                if depth < hops:
                    queue.append((neighbor, depth + 1))
        
        return neighborhood

# Usage example
constructor = GraphConstructor(resolver)

# Process documents
for chunk_id, text in documents:
    entities = resolver.extract_entities_with_context(text, chunk_id)
    relationships = constructor.extract_relationships(text, entities)
    constructor.add_to_graph(chunk_id, text, entities, relationships)

# Query the graph
neighborhood = constructor.get_entity_neighborhood("Dr. Sarah Smith", hops=2)
print(f"Found {len(neighborhood['nodes'])} related entities")
print(f"Spanning {len(neighborhood['chunks'])} document chunks")


#-------- Part 3: Hierarchical Community Detection -----------------------
#Large graphs are hard to search globally. GraphRAG clusters entities into themes.
# GraphRAG uses Leiden community detection to cluster densely connected entities into thematic groups. This enables global search across themes.
from community import community_louvain
from collections import defaultdict

class CommunityAnalyzer:
    """Detect and summarize communities in knowledge graph"""
    
    def __init__(self, graph):
        self.graph = graph
        self.communities = {}
        self.summaries = {}
        
    def detect_communities(self):
        """Apply Leiden/Louvain algorithm for community detection"""
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Detect communities using Louvain algorithm
        partition = community_louvain.best_partition(undirected)
        
        # Group entities by community
        communities = defaultdict(list)
        for entity, comm_id in partition.items():
            communities[comm_id].append(entity)
        
        self.communities = dict(communities)
        return self.communities
    
    def summarize_community(self, community_id, entities):
        """Generate natural language summary of community"""
        
        # Collect all relationships within community
        internal_edges = []
        for source in entities:
            for target in entities:
                if self.graph.has_edge(source, target):
                    edge_data = self.graph.get_edge_data(source, target)
                    for key, attrs in edge_data.items():
                        internal_edges.append({
                            'source': source,
                            'relation': attrs['relation'],
                            'target': target
                        })
        
        # Collect entity types
        entity_info = []
        for entity in entities:
            node_data = self.graph.nodes[entity]
            entity_info.append(f"{entity} ({node_data['type']})")
        
        prompt = f"""Summarize this knowledge community:

Community {community_id}:

Entities: {', '.join(entity_info)}

Key Relationships:
{chr(10).join([f"- {e['source']} {e['relation']} {e['target']}" for e in internal_edges[:20]])}

Provide a 2-3 sentence summary describing:
1. The main theme connecting these entities
2. The domain or topic area
3. Key relationships and patterns"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        summary = response.content[0].text
        self.summaries[community_id] = {
            'summary': summary,
            'size': len(entities),
            'entities': entities,
            'edge_count': len(internal_edges)
        }
        
        return summary
    
    def build_hierarchical_summaries(self):
        """Generate multi-level summaries"""
        
        communities = self.detect_communities()
        
        # Level 1: Individual community summaries
        for comm_id, entities in communities.items():
            self.summarize_community(comm_id, entities)
        
        # Level 2: Meta-communities (cluster of communities)
        if len(communities) > 5:
            # Build community similarity graph
            comm_similarity = nx.Graph()
            for c1 in communities:
                for c2 in communities:
                    if c1 >= c2:
                        continue
                    
                    # Measure inter-community edges
                    cross_edges = sum(
                        1 for e1 in communities[c1] for e2 in communities[c2]
                        if self.graph.has_edge(e1, e2) or self.graph.has_edge(e2, e1)
                    )
                    
                    if cross_edges > 0:
                        comm_similarity.add_edge(c1, c2, weight=cross_edges)
            
            # Detect meta-communities
            meta_partition = community_louvain.best_partition(comm_similarity)
            
            meta_communities = defaultdict(list)
            for comm_id, meta_id in meta_partition.items():
                meta_communities[meta_id].append(comm_id)
            
            # Summarize meta-communities
            for meta_id, community_ids in meta_communities.items():
                all_summaries = [self.summaries[cid]['summary'] for cid in community_ids]
                
                meta_prompt = f"""Synthesize these related community summaries into a high-level theme:

{chr(10).join([f"Community {i}: {s}" for i, s in zip(community_ids, all_summaries)])}

Provide a 2-3 sentence synthesis."""

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=400,
                    messages=[{"role": "user", "content": meta_prompt}]
                )
                
                self.summaries[f"meta_{meta_id}"] = {
                    'summary': response.content[0].text,
                    'sub_communities': community_ids,
                    'type': 'meta'
                }
        
        return self.summaries

# Usage
analyzer = CommunityAnalyzer(constructor.graph)
summaries = analyzer.build_hierarchical_summaries()

print(f"Detected {len(analyzer.communities)} communities")
for comm_id, data in summaries.items():
    if not str(comm_id).startswith('meta'):  # Skip meta-communities
        print(f"\nCommunity {comm_id} ({data['size']} entities):")
        print(f"  {data['summary']}")



#---------- Part 4 Prepare RAG index ---------------------------
#Combine all three search algorithm:
#SQLite FTS5 → keyword search
#FAISS → semantic embeddings
#NetworkX graph → entity relationships

from dataclasses import dataclass
from typing import Optional
import sqlite3
import faiss
import pickle

@dataclass
class DocumentVersion:
    """Track document versions for consistent updates"""
    doc_id: str
    version: int
    chunk_ids: list
    entity_ids: list
    update_timestamp: float

class SynchronizedIndexManager:
    """Manage synchronized updates across text, vector, and graph indexes"""
    
    def __init__(self, db_path="graphrag.db"):
        # Text index (SQLite FTS5)
        self.text_conn = sqlite3.connect(db_path)
        self.text_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts 
            USING fts5(chunk_id, text, doc_id)
        """)
        
        # Vector index (FAISS)
        self.vector_dim = 1536  # text-embedding-3-small dimension
        self.vector_index = faiss.IndexFlatL2(self.vector_dim)
        self.chunk_id_to_vector_idx = {}
        
        # Graph index (NetworkX persisted)
        self.graph_constructor = None  # Will be injected
        
        # Version tracking
        self.versions = {}
        
    def atomic_update(self, doc_id, new_chunks, new_embeddings):
        """Atomically update all three indexes"""
        
        version = self.versions.get(doc_id, DocumentVersion(doc_id, 0, [], [], 0))
        new_version = version.version + 1
        
        try:
            # Step 1: Remove old data
            if version.chunk_ids:
                # Remove from text index
                placeholders = ','.join('?' * len(version.chunk_ids))
                self.text_conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                    version.chunk_ids
                )
                
                # Remove from vector index (mark as deleted)
                for chunk_id in version.chunk_ids:
                    if chunk_id in self.chunk_id_to_vector_idx:
                        # FAISS doesn't support deletion, rebuild periodically
                        pass
                
                # Remove from graph (disconnect old entities)
                for entity_id in version.entity_ids:
                    if self.graph_constructor.graph.has_node(entity_id):
                        # Keep node but remove edges from this doc
                        edges_to_remove = [
                            (u, v, k) for u, v, k, d in 
                            self.graph_constructor.graph.edges(entity_id, keys=True, data=True)
                            if d.get('chunk_id') in version.chunk_ids
                        ]
                        for u, v, k in edges_to_remove:
                            self.graph_constructor.graph.remove_edge(u, v, k)
            
            # Step 2: Add new data
            new_chunk_ids = []
            new_entity_ids = []
            
            for i, (chunk_text, embedding) in enumerate(zip(new_chunks, new_embeddings)):
                chunk_id = f"{doc_id}_chunk_{new_version}_{i}"
                new_chunk_ids.append(chunk_id)
                
                # Add to text index
                self.text_conn.execute(
                    "INSERT INTO chunks_fts VALUES (?, ?, ?)",
                    (chunk_id, chunk_text, doc_id)
                )
                
                # Add to vector index
                vector_idx = self.vector_index.ntotal
                self.vector_index.add(embedding.reshape(1, -1))
                self.chunk_id_to_vector_idx[chunk_id] = vector_idx
                
                # Extract and add to graph
                entities = self.graph_constructor.resolver.extract_entities_with_context(
                    chunk_text, chunk_id
                )
                relationships = self.graph_constructor.extract_relationships(
                    chunk_text, entities
                )
                self.graph_constructor.add_to_graph(
                    chunk_id, chunk_text, entities, relationships
                )
                
                new_entity_ids.extend([
                    self.graph_constructor.resolver.get_canonical_form(e['surface_form'])
                    for e in entities
                ])
            
            # Step 3: Commit transaction
            self.text_conn.commit()
            
            # Update version tracking
            import time
            self.versions[doc_id] = DocumentVersion(
                doc_id, new_version, new_chunk_ids, 
                list(set(new_entity_ids)), time.time()
            )
            
            return True
            
        except Exception as e:
            # Rollback on failure
            self.text_conn.rollback()
            print(f"Update failed: {e}")
            return False
    
    def query_all_indexes(self, query_text, query_embedding, k=5):
        """Query across all three indexes with fusion"""
        
        results = {
            'text_matches': [],
            'vector_matches': [],
            'graph_matches': []
        }
        
        # Text search (keyword)
        cursor = self.text_conn.execute(
            "SELECT chunk_id, text FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            (query_text, k)
        )
        results['text_matches'] = [
            {'chunk_id': row[0], 'text': row[1], 'score': 1.0}
            for row in cursor.fetchall()
        ]
        
        # Vector search (semantic)
        if self.vector_index.ntotal > 0:
            distances, indices = self.vector_index.search(
                query_embedding.reshape(1, -1), k
            )
            reverse_map = {v: k for k, v in self.chunk_id_to_vector_idx.items()}
            results['vector_matches'] = [
                {
                    'chunk_id': reverse_map.get(idx, f'unknown_{idx}'),
                    'score': 1 / (1 + dist)
                }
                for dist, idx in zip(distances[0], indices[0])
                if idx < len(reverse_map)
            ]
        
        # Graph search (entities mentioned in query)
        query_entities = self.graph_constructor.resolver.extract_entities_with_context(
            query_text, "query"
            )
        
        for entity in query_entities:
            
            # Get canonical entities.
            canonical = self.graph_constructor.resolver.get_canonical_form(
                entity['surface_form']
                )
            
            # Get neighboring entities(2 neighbors) from the canonical entities.
            neighborhood = self.graph_constructor.get_entity_neighborhood(
                canonical, hops=2
                )
            
            if neighborhood:
                results['graph_matches'].extend([
                    {
                        'chunk_id': chunk_id,
                        'score': 0.9,
                        'entity': canonical
                    }
                    for chunk_id in neighborhood['chunks']
                ])
        
        # Fusion: combine scores
        all_chunks = {}
        for source, matches in results.items():
            weight = {'text_matches': 0.2, 'vector_matches': 0.4, 'graph_matches': 0.4}[source]
            for match in matches:
                chunk_id = match['chunk_id']
                score = match['score'] * weight
                
                if chunk_id not in all_chunks:
                    all_chunks[chunk_id] = {
                        'chunk_id': chunk_id,
                        'total_score': 0,
                        'sources': []
                    }
                
                all_chunks[chunk_id]['total_score'] += score
                all_chunks[chunk_id]['sources'].append(source)
        
        # Sort by fused score
        ranked = sorted(
            all_chunks.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )
        
        return ranked[:k]


#----------- Test your entity resolver independently before building the graph:
def test_entity_resolution(resolver, test_cases):
    """Validate entity resolution accuracy"""
    
    correct = 0
    total = 0
    
    for surface_form1, surface_form2, should_match in test_cases:
        # Extract with context
        entity1 = {'surface_form': surface_form1, 'type': 'Person', 'context': '', 'features': {}}
        entity2 = {'surface_form': surface_form2, 'type': 'Person', 'context': '', 'features': {}}
        
        similarity = resolver.compute_entity_similarity(entity1, entity2)
        predicted_match = similarity >= 0.75
        
        if predicted_match == should_match:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

# Test cases: (surface_form1, surface_form2, should_match)
test_cases = [
    ("Dr. Sarah Smith", "Dr. S. Smith", True),
    ("Dr. Sarah Smith", "Dr. Michael Smith", False),
    ("Microsoft Corp", "Microsoft Corporation", True),
    ("Apple Inc", "Apple Computer", True),
    ("John Miller", "John Miller", True),  # Same name, need context
]

accuracy = test_entity_resolution(resolver, test_cases)
print(f"Entity resolution accuracy: {accuracy:.2%}")



#--------- Tip: Hybrid search ----------------------------
class AdaptiveGraphRAG:
    def __init__(self):
        self.full_graphrag = None  # Pre-computed summaries
        self.lazy_graphrag = None  # On-demand summaries
        self.query_cache = {}
        
    def route_query(self, query):
        # Check cache first
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Determine if query needs global search
        if self.is_global_query(query):
            # Check if we have pre-computed summaries
            if self.full_graphrag.has_summaries():
                result = self.full_graphrag.global_search(query)
            else:
                # Fall back to LazyGraphRAG
                result = self.lazy_graphrag.global_search(query)
        else:
            # Local search is same for both
            result = self.full_graphrag.local_search(query)
        
        # Cache result
        self.query_cache[query] = result
        return result



