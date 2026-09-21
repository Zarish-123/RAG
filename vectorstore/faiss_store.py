import faiss
import numpy as np

from core.interfaces.vector_store import VectorStore


class FAISSStore(VectorStore):

    def __init__(self, dimension):

        self.dimension = dimension

        # Inner Product + normalized vectors
        # = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)

        # Store documents
        self.documents = []

        # Store normalized embeddings
        # These will also be used by MMR
        self.embeddings = None

    def add_documents(self, documents, embeddings):

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        # Normalize document embeddings
        faiss.normalize_L2(vectors)

        # Add vectors to FAISS
        self.index.add(vectors)

        # Keep normalized vectors for MMR
        self.embeddings = vectors

        # Store documents
        self.documents.extend(documents)

    def search(self, query_embedding, k=5):

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        # Normalize query embedding
        faiss.normalize_L2(query_vector)

        # Search FAISS
        scores, indices = self.index.search(
            query_vector,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index != -1:

                results.append(
                    {
                        "document": self.documents[index],
                        "score": float(score)
                    }
                )

        return results