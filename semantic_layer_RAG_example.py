

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

# Semantic layer using Vector DB.

semantic_chunks = [

    # =========================
    # METRICS(These drive aggregation logic.)
    # =========================
    {
        "type": "metric",
        "name": "total_revenue",
        "text": """
Metric: total_revenue
Meaning: Total revenue from all orders
SQL: SUM(orders.total_amount)
Entity: order
Synonyms: revenue, total revenue, sales, total sales, sales amount
"""
    },
    {
        "type": "metric",
        "name": "order_count",
        "text": """
Metric: order_count
Meaning: Total number of orders
SQL: COUNT(orders.order_id)
Entity: order
Synonyms: number of orders, orders count, total orders
"""
    },
    {
        "type": "metric",
        "name": "customer_count",
        "text": """
Metric: customer_count
Meaning: Total number of unique customers
SQL: COUNT(DISTINCT customers.customer_id)
Entity: customer
Synonyms: number of customers, total customers, unique customers
"""
    },
    {
        "type": "metric",
        "name": "average_order_value",
        "text": """
Metric: average_order_value
Meaning: Average revenue per order
SQL: AVG(orders.total_amount)
Entity: order
Synonyms: AOV, avg order value, average sales per order
"""
    },

    # =========================
    # DIMENSIONS(columns)
    # =========================
    {
        "type": "dimension",
        "name": "customer_id",
        "text": """
Dimension: customer_id
Meaning: Unique identifier of a customer
Column: customers.customer_id
Entity: customer
Synonyms: customer id, user id, client id
"""
    },
    {
        "type": "dimension",
        "name": "customer_name",
        "text": """
Dimension: customer_name
Meaning: Full name of the customer
Column: customers.name
Entity: customer
Synonyms: name, customer name, user name
"""
    },
    {
        "type": "dimension",
        "name": "country",
        "text": """
Dimension: country
Meaning: Country where the customer resides
Column: customers.country
Entity: customer
Synonyms: country, region, nation, location, geography
"""
    },
    {
        "type": "dimension",
        "name": "signup_date",
        "text": """
Dimension: signup_date
Meaning: Date when the customer registered
Column: customers.signup_date
Entity: customer
Synonyms: registration date, join date, signup date
"""
    },
    {
        "type": "dimension",
        "name": "order_id",
        "text": """
Dimension: order_id
Meaning: Unique identifier for an order
Column: orders.order_id
Entity: order
Synonyms: order id, transaction id
"""
    },
    {
        "type": "dimension",
        "name": "order_date",
        "text": """
Dimension: order_date
Meaning: Date when the order was placed
Column: orders.order_date
Entity: order
Synonyms: order date, purchase date, transaction date
"""
    },

    # =========================
    # RELATIONSHIPS (JOINS)
    # =========================
    {
        "type": "relationship",
        "name": "customer_orders",
        "text": """
Relationship: customer to order
Meaning: A customer can place multiple orders
Join: customers.customer_id = orders.customer_id
Type: one-to-many
Synonyms: customer orders, user purchases, customer transactions
"""
    },

    # =========================
    # ENTITIES (TABLE)
    # =========================
    {
        "type": "entity",
        "name": "customer",
        "text": """
Entity: customer
Table: customers
Meaning: A person who has registered and can place orders
Grain: 1 row per customer
Important columns: customer_id, name, country, signup_date
Synonyms: user, client, account
"""
    },
    {
        "type": "entity",
        "name": "order",
        "text": """
Entity: order
Table: orders
Meaning: A purchase transaction made by a customer
Grain: 1 row per order
Important columns: order_id, customer_id, order_date, total_amount
Synonyms: purchase, transaction, sale
"""
    },

    # =========================
    # OPTIONAL: QUERY PATTERN (ADVANCED BOOST)
    # =========================
    {
        "type": "query_pattern",
        "name": "revenue_by_dimension",
        "text": """
Pattern: revenue by dimension
Meaning: To calculate revenue grouped by a customer attribute
Steps:
1. Join customers and orders using customers.customer_id = orders.customer_id
2. Use SUM(orders.total_amount)
3. Group by the selected dimension (e.g., country)
"""
    },
    {
        "type": "query_pattern",
        "name": "time_filter_orders",
        "text": """
Pattern: filter orders by time
Meaning: Apply time filtering on order_date
SQL hint: WHERE orders.order_date BETWEEN <start_date> AND <end_date>
Synonyms: last month, this year, recent orders
"""
    }
]


# Save it in FAISS.
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in semantic_chunks]
embeddings = model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_context(query, top_k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)

    results = [semantic_chunks[i] for i in indices[0]]
    return results

def build_prompt(query):
    retrieved = retrieve_context(query)

    context_text = "\n".join([item["text"] for item in retrieved])

    prompt = f"""
You are a data analyst. Generate SQL based on the context.

Context:
{context_text}

User Question:
{query}

SQL:
"""
    return prompt

import openai

def generate_sql(query):
    prompt = build_prompt(query)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response["choices"][0]["message"]["content"]

#====== Example run =================
query = "total revenue by country last month"
sql = generate_sql(query)
print(sql)
