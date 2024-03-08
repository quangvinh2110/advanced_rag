from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

pretrained_model = "sentence-transformers/all-MiniLM-L6-v2"
save_dir = "/home/vinhnq29/Public/advanced_RAG/langchain/model_hubs/sentence-transformers-all-MiniLM-L6-v2"
cache_dir = "/home/vinhnq29/Public/advanced_RAG/langchain/.cache"

snapshot_download(repo_id=pretrained_model, 
                  ignore_patterns=[".msgpack", ".h5",".safetensors", ".onnx","*.tflite"], 
                  local_dir = save_dir,
                  cache_dir = cache_dir)

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)

model.save_pretrained(save_dir, from_pt=True) 
tokenizer.save_pretrained(save_dir)