import uuid
import os
import json
from typing import List, Tuple, Optional, Callable, Any

from ..utils import absoluteFilePaths

from langchain.text_splitter import TextSplitter



def read_corpus_dir(path: str, extension: str) -> List[dict]:
    
    if extension == "text":
        load_file = lambda f : {"text": f.read(), "filepath": f.name}
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
    input_dir: Optional[str] = None,
    extension: Optional[str] = None,
    docs: Optional[List[dict]] = None, 
    output_dir: str = None,
    text_splitter: TextSplitter = None,
    add_chunk_metadata: Callable[[dict, str], dict] = None
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
        
    chunks_text = []
    chunks_metadata = []
    for doc in docs:
        for chunk in text_splitter.split_text(doc["text"]):

            chunks_text.append(chunk)
            
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
            chunks_metadata.append(chunk_metadata)

    return chunks_text, chunks_metadata