import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import MapReduceDocumentsChain
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PayloadSchema, PayloadSchemaType


openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = "https://6f973fc5-fbc1-4866-9aa0-0d28bfe66ffc.eu-west-1-0.aws.cloud.qdrant.io:6333"

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key= qdrant_api_key
)

collection_name = "chatbot_second_collection"
embedding_size = 1536

qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    on_disk=True,
    payload_schema={
        "document_id": PayloadSchema(type=PayloadSchemaType.KEYWORD)
    }
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key )

pdf_paths = [
    "D:\SkillMate_Intern\RAG_Based_ChatBot_Langchain\data\Infinity_Apps_Guide_for_Administrators_v38.pdf"
    # "D:\SkillMate_Intern\RAG_Based_ChatBot_Langchain\data\Brother_software_user_guide.pdf"
]

all_docs = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    all_docs.extend(loader.load())

def summarize_page(page_content):
    prompt = (f"Summarize the following page in 4-5 sentences."
              f"Page content:\n\"\"\"\n{page_content}\n\"\"\"")
    response = llm.invoke(prompt)
    return response.content

documents_to_store = []

for doc in all_docs:
    if doc.metadata['page'] in [2,3]:
        continue
    else:
        page_no = doc.metadata['page']
        source = doc.metadata['source']
        total_pages = doc.metadata['total_pages']
        page_content = doc.page_content
        summary = (summarize_page(doc.page_content))

    documents_to_store.append(
        Document(
            page_content=summary,
            metadata={
                "full_text": page_content,
                "page_no": page_no,
                "source": source,
                "total_pages": total_pages,
                "document_id": source.split("\\")[-1].replace(".pdf", "")
            }
        )
    )

embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key = openai_api_key)

collection_name = "chatbot_collection"

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="chatbot_collection",
    embedding=embedding_fn,
)

vectorstore.add_documents(documents_to_store)
