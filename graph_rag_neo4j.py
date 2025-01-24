# https://ai.plainenglish.io/graphrag-vs-rag-the-ultimate-use-case-3413fb48bbd4

import pandas as pd

# Load CSV files into pandas DataFrames
customers = pd.read_csv('customers.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')

from neo4j import GraphDatabase
import pandas as pd

class Neo4jLoader:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_nodes(self, query, data):
        with self.driver.session() as session:
            for idx, record in data.iterrows():
                session.run(query, **record.to_dict())

    def create_relationships(self, query, data):
        with self.driver.session() as session:
            for idx, record in data.iterrows():
                session.run(query, **record.to_dict())

# Establish your connection to Neo4j
loader = Neo4jLoader(uri="bolt://localhost:7687", username="neo4j", password="*****")

queries = {
    'Customers': "CREATE (c:Customer {Customer_ID: $customer_id, First_Name: $first_name, Last_Name: $last_name, Email: $email, Phone: $phone, City: $city, Country: $country})",
    'Orders': "CREATE (o:Order {Order_ID: $order_id, Customer_ID: $customer_id, Product_ID: $product_id, Quantity: $quantity, Order_Date: $order_date})",
    'Products': "CREATE (p:Product {Product_ID: $product_id, Product_Name: $product_name, Category: $category, Price: $price})",
    }
    
data_map = {
    'Customers': customers,
    'Orders': orders,
    'Products': products
    }

for label, query in queries.items():
    loader.create_nodes(query, data_map[label])
    
relationship_queries = [
    "MATCH (a:Customer), (b:Order) WHERE a.Customer_ID = b.Customer_ID MERGE (a)-[:ORDERED]->(b)", # file a claim
    "MATCH (a:Order), (b:Customer) WHERE a.Customer_ID = b.Customer_ID MERGE (a)-[:HAS_BEEN_ORDERED]->(b)", # file a claim
    "MATCH (a:Product), (b:Order) WHERE a.Product_ID = b.Product_ID MERGE (a)-[:IS_ORDERED_IN]->(b)"
    ]
    
relationship_data = {
    relationship_queries[0]: orders[['customer_id','order_id']],
    relationship_queries[1]: orders[['order_id', 'customer_id']],
    relationship_queries[2]: orders[['order_id', 'product_id']],
    }
    
for n, (query, data) in enumerate(relationship_data.items()):
    loader.create_relationships(query, data)
  
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="customer_order",
)

graph.refresh_schema()

cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement
If you need to divide numbers, make sure to
filter the denominator to be non zero.
The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
    )

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a users natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.
Query Results:
{context}
Question:
{question}
If the provided information is empty, say you don't know the answer.
Empty information looks like this: []
If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.
Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.
Helpful Answer:
"""

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOllama(model='codellama'), #ChatOpenAI()#,
    qa_llm=ChatOllama(model='llama3'),
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)

hospital_cypher_chain.invoke("What are the product(s) that Susan ordered?")
hospital_cypher_chain.invoke("How many different article were sold in April 2023 ?")


qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)