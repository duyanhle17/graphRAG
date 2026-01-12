from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import logging

logger = logging.getLogger(__name__)

class TokenSplitter:
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            self.enc = tiktoken.encoding_for_model(cfg.tiktoken_model_name)
        except KeyError:
            self.enc = tiktoken.get_encoding("cl100k_base")

        self.chunk_size = cfg.chunk_token_size
        self.overlap = cfg.chunk_overlap_token_size

    def chunk(self, text: str) -> List[str]:
        tokens = self.enc.encode(text)
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(tokens), step):
            sub = tokens[i:i + self.chunk_size]
            if sub:
                chunks.append(self.enc.decode(sub))
        return chunks


def char_splitter(text: str, size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return [d.page_content for d in splitter.create_documents([text])]
