
import re
import numpy as np

from core.interfaces.retriever import Retriever


class SimilarityRetriever(Retriever):

    def __init__(
        self,
        vector_store,
        embedder,
        mmr_lambda=0.70
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.mmr_lambda = mmr_lambda

    # --------------------------------------------------
    # Detect definition query
    # --------------------------------------------------

    def is_definition_query(self, query):

        query = query.lower().strip()

        definition_patterns = [
            "what is",
            "what are",
            "what does",
            "stands for",
            "meaning of",
            "define",
            "definition of"
        ]

        return any(
            pattern in query
            for pattern in definition_patterns
        )

    # --------------------------------------------------
    # Detect application / use-case query
    # --------------------------------------------------

    def is_application_query(self, query):

        query = query.lower().strip()

        application_patterns = [
            "application",
            "applications",
            "use case",
            "use cases",
            "uses",
            "used for",
            "where is",
            "where are",
            "how is",
            "how are",
            "benefits",
            "advantages"
        ]

        return any(
            pattern in query
            for pattern in application_patterns
        )

    # --------------------------------------------------
    # Definition score
    # --------------------------------------------------

    def calculate_definition_score(
        self,
        query,
        content
    ):

        content_lower = content.lower()

        score = 0.0

        # Exact canonical expression
        if re.search(
            r"retrieval[- ]augmented generation\s*\(\s*rag\s*\)",
            content_lower
        ):
            score += 2.0

        # Full RAG expansion
        if (
            "rag" in content_lower
            and "retrieval-augmented generation"
            in content_lower
        ):
            score += 0.8

        # Definition language
        definition_phrases = [
            "is a",
            "is an",
            "refers to",
            "stands for",
            "defined as",
            "definition"
        ]

        for phrase in definition_phrases:

            if phrase in content_lower:
                score += 0.5

        return score

    # --------------------------------------------------
    # Application / use-case score
    # --------------------------------------------------

    def calculate_application_score(
        self,
        content
    ):

        content_lower = content.lower()

        score = 0.0

        application_terms = [
            "application",
            "applications",
            "use case",
            "use cases",
            "used for",
            "used in",
            "use of",
            "benefit",
            "benefits",
            "advantage",
            "advantages",
            "customer service",
            "financial sector",
            "healthcare",
            "health care",
            "education",
            "legal",
            "regulatory compliance",
            "market intelligence",
            "personalized"
        ]

        for term in application_terms:

            if term in content_lower:
                score += 0.30

        return min(score, 2.0)

    # --------------------------------------------------
    # Reference score / penalty
    # --------------------------------------------------

    def calculate_reference_penalty(
        self,
        content
    ):

        content_lower = content.lower()

        reference_terms = [
            "references",
            "openreview.net",
            "nexla.com",
            "arxiv.org",
            "doi.org",
            "retrieved from",
            "http://",
            "https://"
        ]

        matches = 0

        for term in reference_terms:

            if term in content_lower:
                matches += 1

        if matches == 0:
            return 0.0

        # Strong penalty for reference-heavy chunks
        return min(
            3.0,
            1.5 + (matches - 1) * 0.30
        )

    # --------------------------------------------------
    # Find canonical definition
    # --------------------------------------------------

    def find_canonical_definition(
        self,
        documents,
        query
    ):

        if not self.is_definition_query(query):
            return None

        query_lower = query.lower()

        if "rag" not in query_lower:
            return None

        candidates = []

        for document in documents:

            content_lower = document.content.lower()

            if re.search(
                r"retrieval[- ]augmented generation\s*\(\s*rag\s*\)",
                content_lower
            ):

                page = document.metadata.get(
                    "page",
                    999999
                )

                chunk_id = document.metadata.get(
                    "chunk_id",
                    999999
                )

                candidates.append(
                    (
                        page,
                        chunk_id,
                        document
                    )
                )

        if not candidates:
            return None

        # Earliest canonical occurrence
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1]
            )
        )

        return candidates[0][2]

    # --------------------------------------------------
    # MMR
    # --------------------------------------------------

    def apply_mmr(
        self,
        candidates,
        k
    ):

        if not candidates:
            return []

        if k <= 0:
            return []

        if len(candidates) <= k:

            return [
                item["document"]
                for item in candidates
            ]

        embeddings = self.vector_store.embeddings

        scores = np.array(
            [
                item["score"]
                for item in candidates
            ],
            dtype="float32"
        )

        if scores.max() != scores.min():

            normalized_scores = (
                scores - scores.min()
            ) / (
                scores.max() - scores.min()
            )

        else:

            normalized_scores = np.ones(
                len(scores)
            )

        # Find FAISS index for every candidate
        candidate_indices = []

        for item in candidates:

            document = item["document"]

            try:

                index = (
                    self.vector_store.documents.index(
                        document
                    )
                )

            except ValueError:

                index = -1

            candidate_indices.append(index)

        selected_positions = []

        while len(selected_positions) < k:

            best_position = None
            best_mmr_score = -float("inf")

            for position in range(
                len(candidates)
            ):

                if position in selected_positions:
                    continue

                relevance = float(
                    normalized_scores[position]
                )

                if not selected_positions:

                    redundancy = 0.0

                else:

                    current_index = (
                        candidate_indices[position]
                    )

                    if current_index == -1:

                        redundancy = 0.0

                    else:

                        current_embedding = (
                            embeddings[current_index]
                        )

                        similarities = []

                        for selected_position in (
                            selected_positions
                        ):

                            selected_index = (
                                candidate_indices[
                                    selected_position
                                ]
                            )

                            if selected_index == -1:
                                continue

                            selected_embedding = (
                                embeddings[selected_index]
                            )

                            similarity = float(
                                np.dot(
                                    current_embedding,
                                    selected_embedding
                                )
                            )

                            similarities.append(
                                similarity
                            )

                        redundancy = (
                            max(similarities)
                            if similarities
                            else 0.0
                        )

                mmr_score = (
                    self.mmr_lambda * relevance
                    -
                    (1 - self.mmr_lambda)
                    * redundancy
                )

                if mmr_score > best_mmr_score:

                    best_mmr_score = mmr_score
                    best_position = position

            if best_position is None:
                break

            selected_positions.append(
                best_position
            )

        return [
            candidates[position]["document"]
            for position in selected_positions
        ]

    # --------------------------------------------------
    # Main retrieval
    # --------------------------------------------------

    def retrieve(
        self,
        query,
        k=5
    ):

        # ----------------------------------------------
        # Step 1: Query embedding
        # ----------------------------------------------

        query_embedding = self.embedder.embed(
            query
        )

        # ----------------------------------------------
        # Step 2: Semantic candidates
        # ----------------------------------------------

        candidate_k = max(
            k * 5,
            30
        )

        semantic_results = (
            self.vector_store.search(
                query_embedding,
                candidate_k
            )
        )

        candidates = []

        query_lower = query.lower().strip()

        query_words = set(
            re.findall(
                r"\b\w+\b",
                query_lower
            )
        )

        definition_query = (
            self.is_definition_query(query)
        )

        application_query = (
            self.is_application_query(query)
        )

        # ----------------------------------------------
        # Step 3: Score candidates
        # ----------------------------------------------

        for result in semantic_results:

            document = result["document"]

            semantic_score = float(
                result["score"]
            )

            content_lower = (
                document.content.lower()
            )

            # Definition score
            definition_score = (
                self.calculate_definition_score(
                    query,
                    document.content
                )
            )

            # Application score
            application_score = (
                self.calculate_application_score(
                    document.content
                )
            )

            # Reference penalty
            reference_penalty = (
                self.calculate_reference_penalty(
                    document.content
                )
            )

            # ------------------------------------------
            # Exact query phrase
            # ------------------------------------------

            exact_query_bonus = 0.0

            if query_lower in content_lower:

                exact_query_bonus = 1.0

            # ------------------------------------------
            # Keyword overlap
            # ------------------------------------------

            content_words = set(
                re.findall(
                    r"\b\w+\b",
                    content_lower
                )
            )

            overlap = len(
                query_words
                &
                content_words
            )

            keyword_score = min(
                overlap * 0.10,
                0.50
            )

            # ------------------------------------------
            # Query-specific scoring
            # ------------------------------------------

            final_score = (
                semantic_score
                + keyword_score
                + exact_query_bonus
            )

            # Definition question
            if definition_query:

                final_score += (
                    definition_score
                )

            # Application question
            elif application_query:

                final_score += (
                    application_score
                )

                # Definition-only chunks should not
                # dominate an application question.
                final_score -= (
                    definition_score * 0.30
                )

            # Normal question
            else:

                final_score += (
                    definition_score * 0.20
                )

            # References are generally poor
            # answer sources.
            final_score -= reference_penalty

            candidates.append(
                {
                    "document": document,
                    "score": final_score,
                    "semantic": semantic_score,
                    "definition": definition_score,
                    "application": application_score,
                    "reference_penalty": reference_penalty
                }
            )

        # ----------------------------------------------
        # Step 4: Sort candidates
        # ----------------------------------------------

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # ----------------------------------------------
        # Step 5: Canonical definition priority
        # ----------------------------------------------

        canonical_document = (
            self.find_canonical_definition(
                self.vector_store.documents,
                query
            )
        )

        final_documents = []

        if canonical_document is not None:

            # Canonical definition MUST be first
            final_documents.append(
                canonical_document
            )

            remaining_candidates = [
                item
                for item in candidates
                if item["document"]
                is not canonical_document
            ]

            remaining_documents = (
                self.apply_mmr(
                    remaining_candidates,
                    k - 1
                )
            )

            final_documents.extend(
                remaining_documents
            )

        else:

            final_documents = self.apply_mmr(
                candidates,
                k
            )

        # ----------------------------------------------
        # Debug output
        # ----------------------------------------------

        print(
            "\n==================================="
        )

        print(
            "Query-Aware Hybrid + MMR Retrieval"
        )

        print(
            "==================================="
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Definition Query: {definition_query}"
        )

        print(
            f"Application Query: {application_query}"
        )

        print(
            "\nSelected Documents:"
        )

        for i, document in enumerate(
            final_documents,
            start=1
        ):

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                "Unknown"
            )

            print(
                f"{i}. "
                f"Page: {page} | "
                f"Chunk: {chunk_id}"
            )

        print(
            "===================================\n"
        )

        return final_documents

