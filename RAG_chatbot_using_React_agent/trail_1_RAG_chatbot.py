from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate           # _____________________________________________________
from langchain_openai import ChatOpenAI                # |A simple chatbot Using only LLM model to get answer |
import os                                              # |____________________________________________________|

loader = PyPDFLoader("../data/bank_faq.pdf")
docs = loader.load()
content = '\n'.join([doc.page_content for doc in docs])

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key)

template = """
    You are a helpful assistant for answering questions based on the bank FAQ below.

    FAQ:
    {faq}

    Question: {question}
    Answer:
    """
prompt = PromptTemplate.from_template(template)

while True:
    print("Enter your Question here or enter 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        break
    final_prompt = prompt.format(faq=content, question=user_input)
    response = llm.invoke(final_prompt)
    print(f"BOT : {response.content}")