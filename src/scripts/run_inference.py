from src.rag.pipeline import SimpleGraphRAG

if __name__ == "__main__":
    rag = SimpleGraphRAG()

    corpus = "data/corpus/medical.json"
    questions = "data/questions/medical_questions.json"

    text = rag.load_json_and_concat(corpus)
    rag.chunk_text(text)
    rag.build_embeddings_and_index()
    rag.build_kg(text)

    rag.answer_question_llm("What is basal cell carcinoma?")
