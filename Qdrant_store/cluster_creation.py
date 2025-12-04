import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

openai_api_key = os.getenv("OPENAI_API_KEY")
# qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.68ZUuPmNj55gFY2EqevFFIDSMa6cedmbvZFDnIUaffY"
qdrant_url = "https://ee0c1f20-95c1-43b4-b713-4add293f6841.eu-west-1-0.aws.cloud.qdrant.io"


collection_name = "brother_software_collection"
embedding_size = 1536  # For OpenAI text-embedding-3-small

# Qdrant client (disable version compatibility warning if needed)
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    check_compatibility=False
)

# Modern existence & deletion flow
if qdrant_client.collection_exists(collection_name):
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"Warning: Failed to delete existing collection: {e}")

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
)
print(f"Collection created: {collection_name}")