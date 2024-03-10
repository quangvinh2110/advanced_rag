import uuid
import copy
import os
import json
from typing import List, Tuple, Optional, Callable

from ..utils import absoluteFilePaths

from langchain.text_splitter import TokenTextSplitter



def read_corpus_dir(path: str, extension: str) -> List[dict]:
    
    if extension == "text":
        load_file = lambda f : {"text": f.read()}
    elif extension == "json":
        load_file = lambda f : json.loads(f.read())
    else:
        raise ValueError(
            "Only support `text` and `json` file"
        )

    docs = []
    
    for filepath in absoluteFilePaths(path):
        with open(filepath) as f:
            sample = load_file(f)
        sample["filepath"] = filepath
        docs.append(sample)
        
    return docs
        

def docs_chunking(
    input_dir: Optional[str],
    extension: Optional[str],
    docs: Optional[List[dict]], 
    chunk_size: int, 
    chunk_overlap: int,
    output_dir: str
) -> Tuple[List[str], List[dict]]:
    
    output_dir = os.path.abspath(output_dir)
    if not (input_dir or docs):
        raise ValueError(
            "Must have either `input_dir` or `docs`"
        )
    if (not docs) and extension:
        docs = read_corpus_dir(input_dir, extension)
    else:
        raise ValueError(
            "You must provide extension of files in corpus folder"
        )
    for doc in docs:
        doc = copy.deepcopy(doc)
        chunks = []
        metadatas = []
        
        # Split the content into smaller chunks for better manageability.
        for chunk in TokenTextSplitter(chunk_size=chunk_size, 
                                       chunk_overlap=chunk_overlap
                                       ).split_text(doc.pop("text")):
            random_uuid = str(uuid.uuid4())
            chunks.append(chunk)
            
            chunk_file_path = f"{output_dir}/{random_uuid}.json"
            open(chunk_file_path, "w").write(
                json.dumps(chunk, ensure_ascii=False)
            )
            metadatas.append({
                'wiki_file_path': wiki_file_path,
                'wiki_chunk_file_path': wiki_chunk_file_path
            })

        # Add the text chunks and their metadata to the database.
        db.add_texts(texts, metadatas)

def populate_vector_db(DB_PATH="./db/"):
    db = load_vector_db(DB_PATH=DB_PATH)

    # Process each file in the 'wiki/' directory.
    for wiki_file in os.listdir("wiki/"):
        texts = []
        metadatas = []
        
        wiki_file_path  = "wiki/"+wiki_file
        wiki_chunks_dir = "wiki_chunks/"+wiki_file
        os.makedirs(wiki_chunks_dir, exist_ok=True)
       
        # Read the content of the file.
        content = open(wiki_file_path, "r").read()
        # Split the content into smaller chunks for better manageability.
        for chunk in TokenTextSplitter(chunk_size=256).split_text(content):
            random_uuid = str(uuid.uuid4())
            texts.append(chunk)
            
            wiki_chunk_file_path = wiki_chunks_dir+"/"+random_uuid+".txt"
            open(wiki_chunk_file_path, "w").write(chunk)
            metadatas.append({
                'wiki_file_path': wiki_file_path,
                'wiki_chunk_file_path': wiki_chunk_file_path
            })

        # Add the text chunks and their metadata to the database.
        db.add_texts(texts, metadatas)
        
    # Save the components of the database if the directory does not exist.
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    
    cloudpickle.dump(db.docstore._dict, open(DB_PATH+"memoryDocStoreDict.pkl", "wb"))
    cloudpickle.dump(db.index_to_docstore_id, open(DB_PATH+"indexToDocStoreIdDict.pkl", "wb"))
    faiss.write_index(db.index, DB_PATH+"faiss.index")
    
    return db