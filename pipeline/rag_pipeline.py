class RAGPipeline:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question, k=5):

        # Retrieve relevant documents
        documents = self.retriever.retrieve(
            question,
            k
        )

        # If retriever returns None, use an empty list
        if documents is None:
            documents = []

        context_parts = []

        # Prepare context from retrieved documents
        for i, document in enumerate(documents, start=1):

            if document is None:
                continue

            metadata = document.metadata or {}

            page = metadata.get("page", "Unknown")
            chunk_id = metadata.get("chunk_id", "Unknown")
            content = document.content or ""

            context_parts.append(
                f"""
--- Document Information {i} ---

Page: {page}
Chunk: {chunk_id}

Content:
{content}
"""
            )

        # Join all retrieved document content
        context = "\n".join(context_parts)

        # If no context is retrieved
        if not context.strip():
            context = "No relevant information was retrieved from the document."

        prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information provided
in the context below.

Write a proper, detailed, natural paragraph answer.
Do not give only one short sentence.
Explain the answer clearly in approximately 4–6 sentences,
or more if the question requires a detailed explanation.

Use simple and understandable language.
Answer directly and naturally.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not use your general knowledge.
3. Do not invent information.
4. Do not mention context numbers.
5. Do not mention page numbers.
6. Do not mention chunk numbers.
7. Do not mention retrieved sources.
8. Do not write "According to Context".
9. Do not explain where the answer was found.
10. Do not include source references in the answer.
11. Do not answer in a list unless the user specifically asks for a list.
12. If the answer is present in the context, explain it properly in paragraph form.
13. If the answer is not present in the context, write exactly:
"The answer is not available in the provided document."

Context:
{context}

User Question:
{question}

Answer:
"""

        # Generate answer using the language model
        answer = self.llm.generate(prompt)

        # Handle empty response from LLM
        if answer is None:
            answer = "The answer could not be generated."

        return {
            "answer": answer,
            "documents": documents
        }