from typing import (
    Optional,
    Any,
    List
)

from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer
)

from langchain.text_splitter import (
    TextSplitter, 
    Tokenizer, 
    split_text_on_tokens
)


class HfTokenTextSplitter(TextSplitter):
    """Custom huggingface version for langchain.text_splitter.TokenTextSplitter"""
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "vilm/vinallama-7b",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)

        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
            )
        self._tokenizer = tokenizer
        
        
    def split_text(self, text: str) -> List[str]:
        def encode_strip_start_and_stop_token_ids(text: str) -> List[int]:
            return self._encode(text)[1:-1]

        _tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=encode_strip_start_and_stop_token_ids,
        )

        return split_text_on_tokens(text=text, tokenizer=_tokenizer)