
# Sematic Layer (yml file way)
'''

Gemini の回答
When your YAML configuration or semantic manifest grows to
 hundreds or thousands of lines, you hit the "Context Window" 
 problem: the LLM gets overwhelmed, loses focus, or simply 
 runs out of space to process the request.

To handle a massive semantic layer, you need to transition 
from a "Push" model (sending everything) to a "Pull" model 
(sending only what is relevant).

=============== 1. The "Two-Step" Retrieval (RAG for YAML) ============
Instead of feeding the whole file into the prompt, you break 
the YAML into small, manageable chunks (e.g., one chunk per 
table or per business entity) and store them in a Vector Database.

Step 1 (The Search): The LLM looks at the user's question 
(e.g., "What was our retention rate in Tokyo?") and searches 
the Vector DB for "retention," "customer status," and "location."

Step 2 (The Prompt): You only pull the specific YAML 
snippets for the customers and subscriptions entities into the final prompt.


============== 2. Multi-Level Summarization (The "Map" Strategy) ============
Provide the LLM with a "Table of Contents" first, then let 
it ask for the details it needs.

High-Level Map: Send a tiny list of all available Tables/Entities 
and a one-sentence description of each.

Entity Selection: Ask the LLM: "Based on the user's question, which 
3 entities from this list are most likely to contain the answer?"

Detailed Injection: Once the LLM picks orders and products, you 
inject only the full, detailed YAML definitions for those two specific objects.


=============1 3. Modularization (File Splitting) ========================
If you are using tools like dbt or Cube, you shouldn't have 
one giant file anyway. You should split your YAML by 
business domain (e.g., finance.yml, marketing.yml, logistics.yml).

Implementation: Use a "Router" agent. If the user asks about
 "shipping delays," the router directs the query specifically 
 to the logistics context. This keeps the prompt clean and
 reduces the "noise" of unrelated columns.


=========== 4. Semantic Compression ==============================
Often, YAML files are "bloated" with information the LLM doesn't 
actually need for SQL generation (like UI formatting, owner 
names, or update timestamps).

- Pruning: Create a "Lightweight" version of your YAML for the LLM.
- Keep: Column names, Data types, Join logic, and Business descriptions.
- Remove: Personal PII tags, UI display colors, dashboard 
folder paths, and internal metadata.
- Token Optimization: Convert YAML to a more token-efficient format 
like Minified JSON or even a Custom DSL (Domain Specific Language) 
if you need to save every possible bit of space.

======== Comparison =================================
Method	              Best For...	     Complexity
RAG (Vector Search)	  500+ tables 	 High
Two-Step Routing	  50–100 tables	 Medium
Pruning/Compression	  20–50 tables	 Low

'''

#Ontology, knowledge graph way.
'''

========== Tables =============================================
Table1: ent_users (Customers)
Column     Type       Description (The "Meaning")
uid        INT        Unique Identifier for the customer.
reg_dt     DATEThe    date the user joined the platform.
status_cd  INT        1=Active, 2=Churn, 3=Suspended.

Table2: trx_ledger (Transactions)
Column      Type          Description (The "Meaning")
tx_id       INT           Transaction ID.
u_ref       INT           Foreign key mapping to ent_users.uid.
amt_fcy     FLOAT         Amount in Foreign Currency.
base_ccy    VARCHAR       The currency code (e.g., USD, EUR).


========= Sematic layer in yml file ==========================
entities:
  - name: customer
    description: "A person or entity that has registered an account."
    tables: 
      - name: ent_users
        primary_key: uid
    dimensions:
      - name: registration_date
        column: reg_dt
        type: date
      - name: account_status
        column: status_cd
        description: "Status of the user: 1=Active, 2=Churned, 3=Suspended"
        logic: |
          CASE 
            WHEN status_cd = 1 THEN 'Active'
            WHEN status_cd = 2 THEN 'Churned'
            ELSE 'Inactive'
          END

  - name: transaction
    description: "Financial records of purchases made by users."
    tables:
      - name: trx_ledger
        primary_key: tx_id
    dimensions:
      - name: currency
        column: base_ccy
    measures:
      - name: total_spend
        description: "The sum of all transaction amounts in base currency."
        formula: "SUM(amt_fcy)"
      - name: transaction_count
        description: "The total number of unique orders placed."
        formula: "COUNT(tx_id)"

relationships:
  - name: customer_transactions
    type: one_to_many
    join: "customer.uid = transaction.u_ref"
    description: "Connects customers to their specific purchase history."
===========================================================================

'''

import yaml
import networkx as nx

yaml_metadata = """
entities:
  - name: customer
    description: "Contains personal identity and account lifecycle status for registered users."
    tables: 
      - name: ent_users
        description: "Primary production table for user profiles."
        pk: uid
    dimensions:
      - name: account_status
        description: "The current standing of a user. 1 is Active, 2 is Churned."
  
  - name: transaction
    description: "Financial ledger of all successful purchases."
    tables:
      - name: trx_ledger
        description: "Raw transaction logs from the payment gateway."
        pk: tx_id
    measures:
      - name: total_spend
        description: "The monetary value of all transactions, used for calculating ROI."

relationships:
  - name: customer_transactions
    source: customer
    target: transaction
    join_on: "customer.uid = transaction.u_ref"
    description: "Links a customer's identity to their spending history."
"""

def build_rich_semantic_graph(data):
    cfg = yaml.safe_load(data)
    G = nx.DiGraph()

    for ent in cfg['entities']:
        # Add Entity Node with Description
        G.add_node(ent['name'], 
                   type='entity', 
                   desc=ent.get('description', 'No description provided'))
        
        # Add Tables as children
        for tbl in ent.get('tables', []):
            G.add_node(tbl['name'], type='table', desc=tbl.get('description'))
            G.add_edge(ent['name'], tbl['name'], label='implemented_by')

        # Add Dimensions/Measures as children
        for dim in ent.get('dimensions', []) + ent.get('measures', []):
            # If it's a dict (with desc), use it; if string, just name it
            name = dim['name'] if isinstance(dim, dict) else dim
            desc = dim.get('description', '') if isinstance(dim, dict) else ''
            G.add_node(name, type='attribute', desc=desc)
            G.add_edge(ent['name'], name, label='has_attribute')

    # Add Relationships
    for rel in cfg['relationships']:
        G.add_edge(rel['source'], rel['target'], 
                   type='relationship', 
                   join=rel['join_on'],
                   desc=rel.get('description'))
    
    return G

rich_g = build_rich_semantic_graph(yaml_metadata)


# Generating the "LLM Context"
# Now that the graph has descriptions, you can write a helper function that "crawls" the graph to explain the schema to the LLM.
def generate_llm_context(G):
    context = "SYSTEM ONTOLOGY AND SCHEMA:\n"
    for node, data in G.nodes(data=True):
        if data.get('type') == 'entity':
            context += f"\nENTITY: {node}\n- Description: {data['desc']}\n"
            
            # Find associated attributes (dimensions/measures)
            attrs = [n for n in G.neighbors(node) if G.nodes[n].get('type') == 'attribute']
            for a in attrs:
                context += f"  * Attribute: {a} ({G.nodes[a].get('desc')})\n"
                
    context += "\nRELATIONSHIPS:\n"
    for u, v, data in G.edges(data=True):
        if data.get('type') == 'relationship':
            context += f"- {u} can join to {v} via '{data['join']}' ({data['desc']})\n"
            
    return context

def get_relevant_context(G, user_keywords):
    """
    Finds nodes matching keywords and returns them + their immediate neighbors.
    """
    relevant_nodes = set()
    
    # 1. Find 'Seed' nodes based on user keywords
    for node, data in G.nodes(data=True):
        if any(key.lower() in node.lower() or key.lower() in data.get('desc', '').lower() 
               for key in user_keywords):
            relevant_nodes.add(node)
            
            # 2. Add neighbors (The Tables and Relationships needed for this Entity)
            relevant_nodes.update(G.neighbors(node))
            # Also add predecessors (in case the edge direction is reversed)
            relevant_nodes.update(G.predecessors(node))

    # 3. Create a Subgraph of only these nodes
    subgraph = G.subgraph(relevant_nodes)
    
    # 4. Generate the snippet from the subgraph only
    return generate_llm_context(subgraph)

# Example Usage:
# User asks: "What is the total spend for active customers?"
keywords = ["spend", "status", "customer"]
print(f"Keywords:{keywords}")
snippet = get_relevant_context(rich_g, keywords)
print(snippet)

