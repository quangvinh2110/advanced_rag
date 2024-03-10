import uuid
import os
import json
from typing import List, Tuple, Optional, Callable, Any

from ..utils import absoluteFilePaths

from langchain.text_splitter import TextSplitter



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
    output_dir: str,
    text_splitter: TextSplitter,
    add_chunk_metadata: Callable[[Any, Any], dict]
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
        
    chunks = []
    metadatas = []
    for doc in docs:
        for chunk in text_splitter.split_text(doc["text"]):

            chunks.append(chunk)
            
            chunk_metadata = add_chunk_metadata(doc, chunk)
            if "id" in chunk_metadata:
                chunk_id = chunk_metadata["id"]
            else:
                chunk_id = str(uuid.uuid4())
                chunk_metadata["id"] = chunk_id
            chunk_filepath = f"{output_dir}/{chunk_id}.json"
            chunk_metadata["filepath"] = chunk_filepath
            
            open(chunk_filepath, "w").write(
                json.dumps({
                    "text": chunk,
                    **chunk_metadata
                }, ensure_ascii=False, indent=4)
            )
            metadatas.append(chunk_metadata)

    return chunks, metadatas