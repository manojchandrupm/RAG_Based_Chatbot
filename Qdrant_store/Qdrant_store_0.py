import os
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key= qdrant_api_key
)

pdf_paths = [
    # "data/Samsung_washing_Machine_User_Manual.pdf",
    # "data/Samsung_Bespoke_AI_Fridge_user_Manual.pdf",
    "data/Samsung_Q9FNSeries_TV_UserManual.pdf"
]

all_docs = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
chunks = splitter.split_documents(all_docs)
# print(chunks)

embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key = openai_api_key)

collection_name = "chatbot_collection"

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="chatbot_collection",
    embedding=embedding_fn,
)
#
vectorstore.add_documents(chunks)

