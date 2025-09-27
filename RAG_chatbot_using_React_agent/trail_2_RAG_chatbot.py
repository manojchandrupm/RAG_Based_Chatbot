from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate    # ________________________________________________________________________
from langchain_openai import ChatOpenAI         # | 1) A RAG Based chatbot here we load our doc and splitting doc content |
from langchain.chains import RetrievalQA        # | into small chunks and embedding them to store in the vector DB .      |
import os                                       # | 2) the model will get top 3 chunks related to the user query and      |
                                                # | generate the answer from the retrieved chunks .                       |
                                                # |_______________________________________________________________________|
api_key = os.getenv("OPENAI_API_KEY")

loader = PyPDFLoader("../data/Policy Clause New India mediclaim  .pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key= api_key)

INDEX_PATH = "../indexes/my_faiss_index"

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key= api_key )

template = """
    You are an insurance policy assistant specializing in answering questions from the 
    "New India Mediclaim Policy" document.
    
    Rules you must follow:
    1. Answer strictly using only the information in the provided context.  
    2. Do not use outside knowledge, even if you think you know the answer.   
    4. Present the answer clearly, using bullet points or numbered steps if the context contains lists or procedures.  
    5. If the context contains formal text (like company name, address, or legal clauses), preserve that wording in the response.  
     
    Context:
    {context}

    Question: {question}
    Answer:
    """
prompt = PromptTemplate.from_template(template)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

while True:
    print("Enter your Question here or enter 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ["exit","quit","q"] :
        print("Goodbye")
        break
    result = qa.invoke({"query": user_input})
    print("\nBOT:", result["result"])
    print("------------------------------------------------")
