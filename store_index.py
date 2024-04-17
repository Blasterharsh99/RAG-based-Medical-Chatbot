from langchain.vectorstores import Pinecone as PC2
from src.helper import load_pdf,text_split,download_embedding
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
p_key = os.getenv('PINECONE_API_KEY')

extracted_data = load_pdf('data/')
text_chunks = text_split(extracted_data)
embeddings = download_embedding()

pc = Pinecone(api_key = p_key)

index_name = "medical-chatbot"
index = pc.Index(index_name)

docsearch = PC2.from_texts([t.page_content for t in text_chunks],embeddings,index_name=index_name)

