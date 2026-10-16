# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
from dotenv import load_dotenv

from langsmith import traceable  # <-- key import

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_demo

# set the project name over the LangSmith Portal.
os.environ['LANGCHAIN_PROJECT'] = 'Basic RAG APP for single Trace'

load_dotenv()

PDF_PATH = "all_pdfs/Data-Structure-Question-Bank.pdf"  

# ---------- traced setup steps ----------
@traceable(name="load_pdf", tags = ['loader', 'PyPDFLoader'])
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]

@traceable(name="split_documents", tags = ['spliter', 'RecursiveCharacterTextSplitter', 'chunk_size'])
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore", tags = ['Cohere Embedding Model'], metadata = {'embedding_model' : 'embed-english-v3.0'})
def build_vectorstore(splits):
    # initialize the embedding models.
    emb = CohereEmbeddings(
        model = "embed-english-v3.0"
    ) 
    # Create FAISS vector store
    vs = FAISS.from_documents(splits, embedding=emb)
    return vs

# You can also trace a “setup” umbrella span if you want:
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ---------- pipeline ----------
llm = ChatCohere(
    model = 'command-a-03-2025', 
    temperature = 0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "pdf_rag_query_for_single_trace"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)
