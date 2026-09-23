from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import Chunker
from ingestion.ocr_processor import TesseractOCRProcessor
from ingestion.image_extractor import PDFImageExtractor

from embeddings.ollama_embedder import OllamaEmbedder
from vectorstore.faiss_store import FAISSStore
from retrieval.similarity_retriever import SimilarityRetriever
from generation.ollama_llm import OllamaLLM
from pipeline.rag_pipeline import RAGPipeline


def create_rag():

    # -----------------------------
    # 1. Create OCR components
    # -----------------------------

    ocr_processor = TesseractOCRProcessor()

    image_extractor = PDFImageExtractor(
        dpi=200
    )


    # -----------------------------
    # 2. Load PDF
    # -----------------------------

    loader = PDFLoader(
        ocr_processor=ocr_processor,
        image_extractor=image_extractor
    )

    documents = loader.load(
        "data/machine learning.pdf"
    )

    print(f"Loaded {len(documents)} pages")


    # -----------------------------
    # 3. Create chunks
    # -----------------------------

    chunker = Chunker(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = chunker.split(documents)

    print(f"Created {len(chunks)} chunks")


    # -----------------------------
    # 4. Check RAG definition
    # -----------------------------

    print("\n===================================")
    print("Searching for RAG definition...")
    print("===================================")

    definition_found = False

    for i, chunk in enumerate(chunks):

        content = chunk.content.lower()

        if (
            "retrieval-augmented generation" in content
            or "retrieval augmented generation" in content
        ):

            definition_found = True

            print("\nRAG definition found!")
            print(f"Chunk ID: {i}")

            print(
                f"Page: {chunk.metadata.get('page')}"
            )

            print("\nContent:")
            print("-----------------------------")
            print(chunk.content)
            print("-----------------------------")


    if not definition_found:

        print("\nRAG definition was NOT found in chunks.")


    # -----------------------------
    # 5. Create embeddings
    # -----------------------------

    embedder = OllamaEmbedder(
        model="nomic-embed-text"
    )

    embeddings = [
        embedder.embed(chunk.content)
        for chunk in chunks
    ]

    print("\nEmbeddings created")


    # -----------------------------
    # 6. Create FAISS vector store
    # -----------------------------

    vector_store = FAISSStore(
        dimension=len(embeddings[0])
    )

    vector_store.add_documents(
        chunks,
        embeddings
    )

    print("Documents added to FAISS")


    # -----------------------------
    # 7. Create retriever
    # -----------------------------

    retriever = SimilarityRetriever(
        vector_store=vector_store,
        embedder=embedder
    )


    # -----------------------------
    # 8. Create LLM
    # -----------------------------

    llm = OllamaLLM(
        model="llama3.2:3b"
    )


    # -----------------------------
    # 9. Create RAG pipeline
    # -----------------------------

    rag = RAGPipeline(
        retriever=retriever,
        llm=llm
    )

    return rag


# -----------------------------
# Direct testing
# -----------------------------

if __name__ == "__main__":

    rag = create_rag()

    print("\n===================================")
    print("       RAG Chatbot is Ready!")
    print("===================================")

    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() in ["exit", "quit"]:

            print("\nGoodbye!")

            break

        if not question.strip():

            continue

        result = rag.ask(
            question,
            k=4
        )

        print("\nBot:")
        print(result["answer"])

        print("\nRetrieved Sources:")

        for document in result["documents"]:

            print(
                f"\nPage: {document.metadata.get('page')}"
            )

            print(document.content)

        print()

