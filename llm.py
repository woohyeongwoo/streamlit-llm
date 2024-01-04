from langchain.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama

###1. LLM Model
llm = ChatOllama(
    model="mistral:latest",
    temperataure=0.1,
)

###2. Embedding Model
embedding_model = OllamaEmbeddings(model="mistral:latest")

###3. Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def upload_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    # cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


def embed_file(file_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = Chroma.from_documents(docs, embedding_model)

    retriever = vectorstore.as_retriever()
    # retriever = vectorstore.as_retriever(
    #     search_type="mmr", search_kwargs={"k": 1, "fetch_k": 10}
    # )
    return retriever


def execute_chain(retriever, user_message):
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    response = chain.invoke(user_message)
    return response.content
