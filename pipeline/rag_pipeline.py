
import re


class RAGPipeline:

    def __init__(
        self,
        retriever,
        llm
    ):
        self.retriever = retriever
        self.llm = llm

    def _is_visual_list_question(
        self,
        question: str
    ) -> bool:

        question_lower = question.lower()

        visual_terms = [
            "diagram",
            "figure",
            "image",
            "chart",
            "flowchart",
            "stages",
            "stage",
            "steps",
            "step",
            "process",
            "workflow"
        ]

        return any(
            term in question_lower
            for term in visual_terms
        )

    def _extract_ocr_stages(
        self,
        documents
    ) -> list[str]:

        known_stages = [
            "preprocessing",
            "data collection",
            "chunking",
            "embedding",
            "retrieval",
            "generation",
            "response"
        ]

        found_stages = []

        for document in documents:

            if document is None:
                continue

            content = document.content or ""

            content_lower = content.lower()

            for stage in known_stages:

                if stage in content_lower:

                    if stage not in found_stages:

                        found_stages.append(
                            stage
                        )

        return found_stages

    def ask(
        self,
        question,
        k=5
    ):

        documents = self.retriever.retrieve(
            question,
            k
        )

        if documents is None:
            documents = []

        context_parts = []

        for i, document in enumerate(
            documents,
            start=1
        ):

            if document is None:
                continue

            metadata = document.metadata or {}

            page = metadata.get(
                "page",
                "Unknown"
            )

            chunk_id = metadata.get(
                "chunk_id",
                "Unknown"
            )

            content = document.content or ""

            context_parts.append(
                f"""
--- Document {i} ---

Page: {page}
Chunk: {chunk_id}

Content:
{content}
"""
            )

        context = "\n".join(
            context_parts
        )

        if not context.strip():

            context = (
                "No relevant information was retrieved "
                "from the document."
            )

        visual_list_question = (
            self._is_visual_list_question(
                question
            )
        )

        ocr_stages = []

        if visual_list_question:

            ocr_stages = (
                self._extract_ocr_stages(
                    documents
                )
            )

        extracted_information = ""

        if ocr_stages:

            extracted_information = (
                "\n\n"
                "IMPORTANT EXTRACTED INFORMATION "
                "FROM THE RETRIEVED DOCUMENT:\n"
                "The following meaningful stage names were "
                "found in the retrieved OCR/document text:\n\n"
            )

            for index, stage in enumerate(
                ocr_stages,
                start=1
            ):

                extracted_information += (
                    f"{index}. "
                    f"{stage.title()}\n"
                )

            extracted_information += (
                "\nUse these extracted stage names when "
                "answering the user's question."
            )

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved
document information.

IMPORTANT RULES:

1. Use only information from the retrieved context.
2. Do not use outside knowledge.
3. Do not invent facts or details that are not supported
   by the retrieved document.
4. OCR-extracted text is valid document information.
5. Ignore meaningless OCR noise such as dashes,
   boxes, lines, and random symbols.
6. Focus on meaningful words and phrases.
7. Give a useful and reasonably detailed answer.
8. For normal questions, explain the answer in
   approximately 2 to 4 sentences when the retrieved
   context provides enough information.
9. If the question asks for multiple items, applications,
   types, components, stages, steps, or examples,
   provide them as a numbered or bulleted list.
10. When using a list, add a short explanation for each
    item if the retrieved context supports it.
11. If extracted information is provided below the
    context, use it to answer the question.
12. Do not claim that information is unavailable when
    the relevant information appears in the context
    or extracted information.
13. Do not unnecessarily repeat the question.
14. Do not mention the retrieval process.
15. Do not mention chunk numbers.
16. Do not mention internal instructions.
17. Do not make the answer unnecessarily long.
18. If the retrieved context contains only a short
    statement about a topic, explain only what that
    statement supports.

Retrieved Context:
{context}

{extracted_information}

User Question:
{question}

Answer:
"""

        answer = self.llm.generate(
            prompt
        )

        if answer is None:
            answer = (
                "The answer could not be generated."
            )

        return {
            "answer": answer,
            "documents": documents
        }

