from typing import Any, Dict, List

import streamlit as st


class SourceView:

    @staticmethod
    def render(
        sources: List[Dict[str, Any]]
    ) -> None:

        if not sources:
            return

        st.markdown("### 📚 Retrieved Sources")

        for index, source in enumerate(
            sources,
            start=1
        ):

            metadata = source.get(
                "metadata",
                {}
            )

            page = metadata.get(
                "page",
                "N/A"
            )

            chunk_id = metadata.get(
                "chunk_id",
                "N/A"
            )

            st.markdown(
                f"""
                <div class="source-box">

                <b>Source {index}</b><br>

                📄 Page: {page}<br>

                🔹 Chunk: {chunk_id}

                </div>
                """,
                unsafe_allow_html=True
            )