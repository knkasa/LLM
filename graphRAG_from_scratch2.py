from openai import OpenAI
client = OpenAI()

# Extract entities from texts using LLM.
prompt = """
Extract entities and relations as JSON.

Schema:
Entities: Company, Product, Person, Technology
Relations:
- Company DEVELOPS Product
- Product USES Technology
- Person WORKS_AT Company

Text:
OpenAI develops ChatGPT. ChatGPT uses transformer models.
Sam Altman works at OpenAI.
"""

response = client.responses.create(model="gpt-4.1",input=prompt)
print(response.output_text)

'''
{
  "entities": [
    {"id": "OpenAI", "type": "Company"},
    {"id": "ChatGPT", "type": "Product"},
    {"id": "Sam Altman", "type": "Person"},
    {"id": "Transformer", "type": "Technology"}
  ],
  "relations": [
    {"source": "OpenAI", "type": "DEVELOPS", "target": "ChatGPT"},
    {"source": "ChatGPT", "type": "USES", "target": "Transformer"},
    {"source": "Sam Altman", "type": "WORKS_AT", "target": "OpenAI"}
  ]
}
'''

# Save network.  You could use networkx library.
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687",auth=("neo4j", "password"))

def insert_kg(tx):
    tx.run("""
    MERGE (c:Company {name: 'OpenAI'})
    MERGE (p:Product {name: 'ChatGPT'})
    MERGE (t:Technology {name: 'Transformer'})
    MERGE (s:Person {name: 'Sam Altman'})
    MERGE (c)-[:DEVELOPS]->(p)
    MERGE (p)-[:USES]->(t)
    MERGE (s)-[:WORKS_AT]->(c)
    """)

with driver.session() as session:
    session.execute_write(insert_kg)

# Function that can query knowledge graph from user_prompt.
def graph_rag_query(user_question, graph_db, llm):
  
    # Step 1: Extract entities from question (using llm)
    entities = llm.extract_entities(user_question)
    
    # Step 2: Query graph
    context_subgraph = graph_db.query(f"""
        MATCH (e)-[r*1..2]-(related)
        WHERE e.name IN {entities}
        RETURN e, r, related
    """)
    
    # Step 3: Convert to text
    context = subgraph_to_text(context_subgraph)
    
    # Step 4: Generate answer
    return llm.generate(f"Context: {context}\n\nQuestion: {user_question}")

user_question = "What technology does the product developed by OpenAI use?"
response = graph_rag_query(user_question, graph_db, llm)
