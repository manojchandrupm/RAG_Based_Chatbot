import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Set up credentials
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.68ZUuPmNj55gFY2EqevFFIDSMa6cedmbvZFDnIUaffY"
qdrant_url = "https://ee0c1f20-95c1-43b4-b713-4add293f6841.eu-west-1-0.aws.cloud.qdrant.io"


collection_name = "brother_software_collection"
embedding_size = 1536  # For OpenAI text-embedding-3-small

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=60.0,
    check_compatibility=False
)

# Load PDF pages (example)
pdf_paths = [
    r"D:\SkillMate_Intern\RAG_Based_ChatBot_Langchain\data\Brother_software_user_guide.pdf"
]
all_docs = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    all_docs.extend(loader.load())

# Summarization helper function
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key
)

def summarize_page(page_content):
    prompt = (
        f"Summarize the following page in 4-5 sentences.\n"
        f"Page content:\n\"\"\"\n{page_content}\n\"\"\""
    )
    response = llm.invoke(prompt)
    return response.content

# Prepare summarized docs, skipping pages 2 and 3 as per your pattern
documents_to_store = []
for doc in all_docs:
    if doc.metadata['page'] in [2, 3]:
        continue
    summary = summarize_page(doc.page_content)
    # Use rsplit in case source uses / or \ as path separator
    document_id = doc.metadata['source'].rsplit("\\", 1)[-1].replace(".pdf", "")
    documents_to_store.append(
        Document(
            page_content=summary,
            metadata={
                "full_text": doc.page_content,
                "page_no": doc.metadata['page'],
                "source": doc.metadata['source'],
                "total_pages": doc.metadata['total_pages'],
                "document_id": document_id
            }
        )
    )

# Embedding function
embedding_fn = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

# Store in Qdrant
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_fn
)

vectorstore.add_documents(documents_to_store)
print("All documents added successfully!")
