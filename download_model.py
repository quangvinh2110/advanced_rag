from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

pretrained_model = ""
save_dir = ""
cache_dir = ""

snapshot_download(repo_id=pretrained_model, 
                  ignore_patterns=[".msgpack", ".h5",".safetensors", ".onnx","*.tflite"], 
                  local_dir = save_dir,
                  cache_dir = cache_dir)

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)

model.save_pretrained(save_dir, from_pt=True) 
tokenizer.save_pretrained(save_dir)