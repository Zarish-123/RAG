
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

        # These words usually indicate that the user
        # is asking about a specific page, diagram,
        # figure, process, or stages.
        #
        # Therefore, such queries should NOT be treated
        # as definition questions.

        non_definition_terms = [
            "page",
            "diagram",
            "figure",
            "chart",
            "graph",
            "image",
            "illustration",
            "shown",
            "stages",
            "stage",
            "steps",
            "step",
            "process",
            "flow",
            "workflow",
            "listed",
            "mentioned"
        ]

        if any(
            term in query
            for term in non_definition_terms
        ):
            return False

        # More specific definition patterns.
        #
        # IMPORTANT:
        # We do NOT simply use "what is" or "what are"
        # because queries like:
        #
        # "What are the stages on page 5?"
        #
        # are not definition questions.

        definition_patterns = [
            r"\bwhat is\s+(the\s+)?rag\b",
            r"\bwhat is\s+(the\s+)?retrieval[- ]augmented generation\b",
            r"\bwhat does\s+rag\s+mean\b",
            r"\bwhat does\s+rag\s+stand for\b",
            r"\bwhat is\s+chunking\b",
            r"\bwhat is\s+embedding\b",
            r"\bwhat is\s+retrieval\b",
            r"\bwhat is\s+generation\b",
            r"\bwhat is\s+ocr\b",
            r"\bwhat is\s+the\s+meaning of\b",
            r"\bdefine\b",
            r"\bdefinition of\b",
            r"\bmeaning of\b",
            r"\bstands for\b"
        ]

        return any(
            re.search(
                pattern,
                query
            )
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
    # Detect requested page number
    # --------------------------------------------------

    def get_requested_page(self, query):

        query = query.lower().strip()

        patterns = [
            r"\bpage\s*(\d+)\b",
            r"\bp\.\s*(\d+)\b"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query
            )

            if match:

                return int(
                    match.group(1)
                )

        return None

    # --------------------------------------------------
    # Detect diagram / visual query
    # --------------------------------------------------

    def is_visual_reference_query(self, query):

        query = query.lower().strip()

        visual_terms = [
            "diagram",
            "figure",
            "chart",
            "graph",
            "image",
            "illustration",
            "shown",
            "shown in",
            "stages",
            "stage",
            "steps",
            "step",
            "flow",
            "workflow"
        ]

        return any(
            term in query
            for term in visual_terms
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
            r"retrieval[- ]augmented generation\s*\(?\s*rag\s*\)?",
            content_lower
        ):

            score += 2.0

        # Full RAG expansion

        if (
            "rag" in content_lower
            and
            "retrieval-augmented generation"
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

        return min(
            score,
            2.0
        )

    # --------------------------------------------------
    # Reference penalty
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

        return min(
            3.0,
            1.5 + (matches - 1) * 0.30
        )

    # --------------------------------------------------
    # Find canonical RAG definition
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

            content_lower = (
                document.content.lower()
            )

            if re.search(
                r"retrieval[- ]augmented generation\s*\(?\s*rag\s*\)?",
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

        embeddings = (
            self.vector_store.embeddings
        )

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

        # Find vector-store index
        # for every candidate

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

            candidate_indices.append(
                index
            )

        selected_positions = []

        while len(selected_positions) < k:

            best_position = None

            best_mmr_score = -float(
                "inf"
            )

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
                            embeddings[
                                current_index
                            ]
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
                                embeddings[
                                    selected_index
                                ]
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
                    self.mmr_lambda
                    * relevance
                    -
                    (1 - self.mmr_lambda)
                    * redundancy
                )

                if mmr_score > best_mmr_score:

                    best_mmr_score = (
                        mmr_score
                    )

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
        # Step 1: Create query embedding
        # ----------------------------------------------

        query_embedding = (
            self.embedder.embed(
                query
            )
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

        query_lower = (
            query.lower().strip()
        )

        query_words = set(
            re.findall(
                r"\b\w+\b",
                query_lower
            )
        )

        # ----------------------------------------------
        # Query classification
        # ----------------------------------------------

        definition_query = (
            self.is_definition_query(
                query
            )
        )

        application_query = (
            self.is_application_query(
                query
            )
        )

        requested_page = (
            self.get_requested_page(
                query
            )
        )

        visual_query = (
            self.is_visual_reference_query(
                query
            )
        )

        # ----------------------------------------------
        # Step 3: Score semantic candidates
        # ----------------------------------------------

        for result in semantic_results:

            document = result["document"]

            semantic_score = float(
                result["score"]
            )

            content_lower = (
                document.content.lower()
            )

            document_page = (
                document.metadata.get(
                    "page"
                )
            )

            # ------------------------------------------
            # Definition score
            # ------------------------------------------

            definition_score = (
                self.calculate_definition_score(
                    query,
                    document.content
                )
            )

            # ------------------------------------------
            # Application score
            # ------------------------------------------

            application_score = (
                self.calculate_application_score(
                    document.content
                )
            )

            # ------------------------------------------
            # Reference penalty
            # ------------------------------------------

            reference_penalty = (
                self.calculate_reference_penalty(
                    document.content
                )
            )

            # ------------------------------------------
            # Exact query phrase bonus
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
            # Base score
            # ------------------------------------------

            final_score = (
                semantic_score
                + keyword_score
                + exact_query_bonus
            )

            # ------------------------------------------
            # Definition query
            # ------------------------------------------

            if definition_query:

                final_score += (
                    definition_score
                )

            # ------------------------------------------
            # Application query
            # ------------------------------------------

            elif application_query:

                final_score += (
                    application_score
                )

                final_score -= (
                    definition_score
                    * 0.30
                )

            # ------------------------------------------
            # Normal query
            # ------------------------------------------

            else:

                final_score += (
                    definition_score
                    * 0.20
                )

            # ------------------------------------------
            # Page-specific boost
            # ------------------------------------------

            if (
                requested_page is not None
                and
                document_page == requested_page
            ):

                # User explicitly requested this page.

                final_score += 3.0

            # ------------------------------------------
            # Visual / diagram boost
            # ------------------------------------------

            if visual_query:

                visual_content_terms = [
                    "stage",
                    "stages",
                    "step",
                    "steps",
                    "diagram",
                    "figure",
                    "process",
                    "workflow",
                    "data collection",
                    "preprocessing",
                    "chunking",
                    "embedding",
                    "retrieval",
                    "generation",
                    "response"
                ]

                visual_matches = 0

                for term in visual_content_terms:

                    if term in content_lower:

                        visual_matches += 1

                visual_bonus = min(
                    visual_matches * 0.15,
                    1.50
                )

                final_score += (
                    visual_bonus
                )

            # ------------------------------------------
            # Reference penalty
            # ------------------------------------------

            final_score -= (
                reference_penalty
            )

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
        # Step 4:
        # Add requested-page documents if FAISS
        # did not return them
        # ----------------------------------------------

        if requested_page is not None:

            existing_documents = {
                id(item["document"])
                for item in candidates
            }

            for document in (
                self.vector_store.documents
            ):

                document_page = (
                    document.metadata.get(
                        "page"
                    )
                )

                if (
                    document_page
                    == requested_page
                    and
                    id(document)
                    not in existing_documents
                ):

                    content_lower = (
                        document.content.lower()
                    )

                    # Calculate keyword overlap

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

                    # Strong page-specific score

                    page_score = (
                        3.0
                        + keyword_score
                    )

                    # Additional visual boost

                    if visual_query:

                        visual_content_terms = [
                            "stage",
                            "stages",
                            "step",
                            "steps",
                            "diagram",
                            "figure",
                            "process",
                            "workflow",
                            "data collection",
                            "preprocessing",
                            "chunking",
                            "embedding",
                            "retrieval",
                            "generation",
                            "response"
                        ]

                        visual_matches = sum(
                            term in content_lower
                            for term in visual_content_terms
                        )

                        page_score += min(
                            visual_matches * 0.15,
                            1.50
                        )

                    candidates.append(
                        {
                            "document": document,
                            "score": page_score,
                            "semantic": 0.0,
                            "definition": 0.0,
                            "application": 0.0,
                            "reference_penalty": 0.0
                        }
                    )

        # ----------------------------------------------
        # Step 5: Sort candidates
        # ----------------------------------------------

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # ----------------------------------------------
        # Step 6: Canonical definition priority
        # ----------------------------------------------

        canonical_document = (
            self.find_canonical_definition(
                self.vector_store.documents,
                query
            )
        )

        final_documents = []

        if canonical_document is not None:

            # Canonical RAG definition comes first

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

            final_documents = (
                self.apply_mmr(
                    candidates,
                    k
                )
            )

        # ----------------------------------------------
        # Step 7: Debug information
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
            f"Definition Query: "
            f"{definition_query}"
        )

        print(
            f"Application Query: "
            f"{application_query}"
        )

        print(
            f"Requested Page: "
            f"{requested_page}"
        )

        print(
            f"Visual Query: "
            f"{visual_query}"
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

