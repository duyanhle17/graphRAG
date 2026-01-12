import types
# from src.llm.qwen_local import qwen_reasoning
# from llm.qwen_local import qwen_reasoning
from src.llm.qwen_local import qwen_reasoning

import sys
import os
sys.path.append(os.path.abspath("src"))

class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": [[1, 2, 3]]}

    def decode(self, ids, skip_special_tokens=True):
        return "Context\nReasoning: this is a test reasoning"

class DummyModel:
    def generate(self, **kwargs):
        return [[0, 1, 2]]

def test_qwen_reasoning_returns_string():
    tokenizer = DummyTokenizer()
    model = DummyModel()

    out = qwen_reasoning(
        query="What is X?",
        context="X is something.",
        tokenizer=tokenizer,
        model=model
    )

    assert isinstance(out, str)
    assert len(out) > 0
