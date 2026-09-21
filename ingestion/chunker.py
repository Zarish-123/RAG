from core.interfaces.models.document import Document


class Chunker:

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):

        chunks = []

        for document in documents:

            text = document.content.strip()

            if not text:
                continue

            # Split document into paragraphs
            paragraphs = text.split("\n\n")

            current_chunk = ""

            for paragraph in paragraphs:

                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                # If paragraph fits into current chunk
                if len(current_chunk) + len(paragraph) <= self.chunk_size:

                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph

                else:

                    # Save current chunk
                    if current_chunk.strip():

                        chunks.append(
                            Document(
                                content=current_chunk.strip(),
                                metadata={
                                    **document.metadata,
                                    "chunk_id": len(chunks)
                                }
                            )
                        )

                    # Create overlap
                    overlap_text = current_chunk[
                        max(0, len(current_chunk) - self.chunk_overlap):
                    ]

                    current_chunk = overlap_text + "\n\n" + paragraph

                    # If paragraph itself is too large,
                    # split it into smaller pieces
                    while len(current_chunk) > self.chunk_size:

                        chunk_text = current_chunk[:self.chunk_size]

                        chunks.append(
                            Document(
                                content=chunk_text.strip(),
                                metadata={
                                    **document.metadata,
                                    "chunk_id": len(chunks)
                                }
                            )
                        )

                        current_chunk = current_chunk[
                            self.chunk_size - self.chunk_overlap:
                        ]

            # Add remaining text
            if current_chunk.strip():

                chunks.append(
                    Document(
                        content=current_chunk.strip(),
                        metadata={
                            **document.metadata,
                            "chunk_id": len(chunks)
                        }
                    )
                )

        return chunks