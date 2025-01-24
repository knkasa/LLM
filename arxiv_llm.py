# RAG arxiv paper through LLM with Vector storage.
# https://ai.plainenglish.io/docling-transforming-document-processing-for-the-modern-era-99ba230062a6
# For Graph RAG, https://ai.plainenglish.io/graphrag-vs-rag-the-ultimate-use-case-3413fb48bbd4

from docling.document_converter import DocumentConverter

# Initialize the converter
converter = DocumentConverter()

# The AI paper:
papers = "https://arxiv.org/pdf/1706.03762.pdf", # Attention Is All You Need
result = converter.convert(papers)

# Export to markdown for easy reading and processing
markdown_content = result.document.export_to_markdown()
print(f"Successfully processed: {papers}")

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.docstore.document import Document
import pandas as pd


class DocumentExtraction:
    def __init__(self, url) -> None:
        loader = DoclingPDFLoader(file_path=url)
        self.document = loader.load()

    def extract_document(self, chunk_size=500, chunk_overlap=100) -> None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n"])
        self.splitter = text_splitter.split_documents(self.document)


    def save_to_chroma(self):
        db = Chroma.from_documents(
            self.splitter,
            embedding = OllamaEmbeddings(model="nomic-embed-text"),
        )
        self.db = db


    def query_rag(self, query_text):
        results = self.db.similarity_search_with_relevance_scores(query_text, k=3)
        context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template("""Answer my question: {question}""")
        prompt = prompt_template.format(context=context_text, question=query_text)
        model = ChatOllama(model="llama3")
        response_text = model.predict(prompt)
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return formatted_response, response_text
        
URL = "https://arxiv.org/pdf/1706.03762.pdf"
new = DocumentExtraction(url=URL)
new.extract_document()
new.save_to_chroma()

new.query_rag("WHat is the title of the document and who wrote it ??")


