

#Table=customers
#| Column      | Type   | Description                         |
#| ----------- | ------ | ----------------------------------- |
#| customer_id | STRING | Unique identifier for each customer |
#| name        | STRING | Full name of the customer           |
#| signup_date | DATE   | Date when the customer registered   |
#| country     | STRING | Country where the customer lives    |

#2. Table: orders
#| Column       | Type   | Description                      |
#| ------------ | ------ | -------------------------------- |
#| order_id     | STRING | Unique identifier for each order |
#| customer_id  | STRING | Foreign key linking to customers |
#| order_date   | DATE   | Date when the order was placed   |
#| total_amount | FLOAT  | Total value of the order (USD)   |

# (In practice you’d load this via yaml.safe_load, but here we hardcode it)
# You should also put descriptions to each columns.
semantic_layer = {
    "entities": {
        "customer": {
            "table": "customers",
            "primary_key": "customer_id",
            "grain": "1 row per customer"
        },
        "order": {
            "table": "orders",
            "primary_key": "order_id",
            "grain": "1 row per order"
        }
    },

    "relationships": [
        {
            "name": "customer_orders",
            "from_entity": "customer",
            "to_entity": "order",
            "type": "one_to_many",
            "join": {
                "from_column": "customer_id",
                "to_column": "customer_id"
            }
        }
    ],

    "dimensions": [
        {"name": "customer_id", "entity": "customer", "column": "customer_id"},
        {"name": "country", "entity": "customer", "column": "country"},
        {"name": "order_id", "entity": "order", "column": "order_id"},
        {"name": "order_date", "entity": "order", "column": "order_date"}
    ],

    "metrics": [
        {
            "name": "total_revenue",
            "entity": "order",
            "calculation": "SUM(orders.total_amount)"
        }
    ]
}

#=========== Create graph =================================
import networkx as nx

def build_graph(semantic_layer):
    G = nx.DiGraph()

    # --- Add entity nodes ---
    for entity_name, entity_data in semantic_layer["entities"].items():
        G.add_node(
            entity_name,
            type="entity",
            table=entity_data["table"],
            primary_key=entity_data["primary_key"],
            grain=entity_data["grain"]
        )

    # --- Add dimension (column) nodes ---
    for dim in semantic_layer["dimensions"]:
        col_node = f"{dim['entity']}.{dim['column']}"

        G.add_node(
            col_node,
            type="column",
            column=dim["column"]
        )

        # Link entity -> column
        G.add_edge(
            dim["entity"],
            col_node,
            relation="has_column"
        )

    # --- Add metric nodes ---
    for metric in semantic_layer["metrics"]:
        G.add_node(
            metric["name"],
            type="metric",
            formula=metric["calculation"]
        )

        # Link metric -> entity
        G.add_edge(
            metric["name"],
            metric["entity"],
            relation="defined_on"
        )

    # --- Add relationships ---
    for rel in semantic_layer["relationships"]:
        join_str = f"{rel['from_entity']}.{rel['join']['from_column']} = {rel['to_entity']}.{rel['join']['to_column']}"

        G.add_edge(
            rel["from_entity"],
            rel["to_entity"],
            relation=rel["type"],
            join=join_str
        )

    return G

G = build_graph(semantic_layer)

for node, data in G.nodes(data=True):
    print(node, data)
for u, v, data in G.edges(data=True):
    print(f"{u} -> {v} | {data}")

#===== Suppose LLM extracted keywords from user prompt. ==============
# Extract parts of the graph, so LLM can create SQL query.
def get_column_node(G, dimension_name):
    for node, data in G.nodes(data=True):
        if data.get("type") == "column" and node.endswith(f".{dimension_name}"):
            return node
    return None

def get_entity_from_column(G, column_node):
    for u, v, data in G.edges(data=True):
        if v == column_node and data.get("relation") == "has_column":
            return u
    return None

def get_metric_entity(G, metric_name):
    for u, v, data in G.edges(data=True):
        if u == metric_name and data.get("relation") == "defined_on":
            return v
    return None

import networkx as nx

def extract_subgraph(G, extracted):
    nodes_to_include = set()
    edges_to_include = []

    # --- Metrics ---
    for metric in extracted.get("metrics", []):
        nodes_to_include.add(metric)

        entity = get_metric_entity(G, metric)
        if entity:
            nodes_to_include.add(entity)
            edges_to_include.append((metric, entity))

    # --- Dimensions ---
    for dim in extracted.get("dimensions", []):
        col_node = get_column_node(G, dim)

        if col_node:
            nodes_to_include.add(col_node)

            entity = get_entity_from_column(G, col_node)
            if entity:
                nodes_to_include.add(entity)
                edges_to_include.append((entity, col_node))

    # --- Ensure joins between involved entities ---
    entities = [n for n in nodes_to_include if G.nodes[n].get("type") == "entity"]

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if nx.has_path(G, entities[i], entities[j]):
                path = nx.shortest_path(G, entities[i], entities[j])

                for k in range(len(path) - 1):
                    u, v = path[k], path[k+1]
                    nodes_to_include.add(u)
                    nodes_to_include.add(v)
                    edges_to_include.append((u, v))

    # --- Build subgraph ---
    subG = G.subgraph(nodes_to_include).copy()

    return subG

def subgraph_to_context(subG):
    context = []

    # Nodes
    for node, data in subG.nodes(data=True):
        if data["type"] == "metric":
            context.append(f"Metric {node} = {data['formula']}")

        elif data["type"] == "entity":
            context.append(f"Entity {node} uses table {data['table']}")

        elif data["type"] == "column":
            context.append(f"Column {node}")

    # Edges (joins)
    for u, v, data in subG.edges(data=True):
        if "join" in data:
            context.append(f"Join: {data['join']}")

    return "\n".join(context)

# Test
query_entities = {
    "metrics": ["total_revenue"],
    "dimensions": ["country"]
}

subG = extract_subgraph(G, query_entities)
context = subgraph_to_context(subG)

print(context)