import os
import faiss
import cloudpickle
from operator import itemgetter

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.llms.huggingface_text_gen_inference import (
        HuggingFaceTextGenInference,
)
from langchain_community.callbacks import streaming_stdout
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.schema.runnable import RunnableMap    
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from src.retrieval.utils import docs_chunking
from src.prompts import RAG_TEMPLATE


# Set the path for the database directory.
DOCS_DIR = "/home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/docs"
EXTENSION = "text"
CHUNKS_DIR = "/home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/chunks"
INDEX_DIR = "/home/vinhnq29/Public/advanced_RAG/langchain/db/wiki/index"


# Function to load or create a vector database.
# This database will be used for storing and retrieving document embeddings.
def load_vector_db(index_dir: str = None):
    # Initialize variables for the components of the database.
    db = None
    memoryDocStoreDict = {}
    indexToDocStoreIdDict = {}
    
    # Check if the database already exists. If it does, load its components.
    if os.path.exists(index_dir):
        memoryDocStoreDict = cloudpickle.load(open(
            os.path.join(index_dir, "memoryDocStoreDict.pkl"), "rb"
        ))
        indexToDocStoreIdDict = cloudpickle.load(open(
            os.path.join(index_dir, "indexToDocStoreIdDict.pkl"), "rb"
        ))
        index = faiss.read_index(
            os.path.join(index_dir, "faiss.index")
        )
    else:
        # If the database does not exist, create a new FAISS index.
        index = faiss.IndexFlatL2(384)

    # Create the FAISS vector database with the loaded or new components.
    db = FAISS(
        index=index,
        docstore=InMemoryDocstore(memoryDocStoreDict),
        index_to_docstore_id=indexToDocStoreIdDict,
        embedding_function=HuggingFaceEmbeddings(
            model_name='/home/vinhnq29/Public/advanced_RAG/langchain/model_hubs/sentence-transformers-all-MiniLM-L6-v2', 
            model_kwargs={'device': 'cpu'}
        )
    )
    return db

# Function to populate the vector database with documents.
def populate_vector_db():
    
    db = load_vector_db()
    text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=10)

    def add_chunk_metadata(doc: dict, chunk_text: str) -> dict:
        chunk_metadata = {}
        for k, v in doc.items():
            chunk_metadata[f"doc_{k}"] = v
        chunk_metadata.pop("doc_text")
        return chunk_metadata

    # Process each file in the 'wiki/' directory.
    texts, metadatas = docs_chunking(
        input_dir=DOCS_DIR,
        extension=EXTENSION,
        output_dir=CHUNKS_DIR,
        text_splitter=text_splitter,
        add_chunk_metadata=add_chunk_metadata
    )

    # Add the text chunks and their metadata to the database.
    db.add_texts(texts, metadatas)
        
    # Save the components of the database if the directory does not exist.
    index_dir = INDEX_DIR
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    cloudpickle.dump(
        db.docstore._dict, 
        open(os.path.join(index_dir, "memoryDocStoreDict.pkl"), "wb")
    )
    cloudpickle.dump(
        db.index_to_docstore_id, 
        open(os.path.join(index_dir, "indexToDocStoreIdDict.pkl"), "wb")
    )
    faiss.write_index(
        db.index, 
        os.path.join(index_dir, "faiss.index")
    )
    
    return db

# Function to configure and retrieve a large language model from Hugging Face.
def get_llm():
    
    # Define the model name and retrieve the necessary token for authentication.
    callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
    llm = HuggingFaceTextGenInference(
        inference_server_url="http://localhost:8010/",
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        callbacks=callbacks,
        streaming=True
    )

    # Load the tokenizer from Hugging Face with the specified configurations.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = ""
    chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)
    return chat_model

# Function to format a list of documents into a single string.
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

# Function to ask a question and receive an answer using the large language model and the document database.
def ask(q):
    # Define a template for the prompt to be used with the large language model.
    rag_prompt = PromptTemplate.from_template(RAG_TEMPLATE)
    base_prompt = None

    llm = get_llm()

    # Create chains of operations to process the question.   
    base_chain = (
        
    )
    rag_chain = (
        {"documents": db.as_retriever(), "question": RunnablePassthrough()}
        | {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain of operations with the question.
    response = rag_chain_with_source.invoke(q)
    print(response["answer"])
    for doc in response["documents"]:
        print(doc['wiki_chunk_file_path'])
    

# Main execution block: populate and load the vector database, then use it to answer a sample question.
if __name__=="__main__":
    if not os.path.exists(INDEX_DIR):
        db = populate_vector_db()
    db = load_vector_db(index_dir=INDEX_DIR)
    ask("What is the capital of NJ?")
