from typing import Optional

import streamlit as st


class WelcomeView:

    @staticmethod
    def render() -> Optional[str]:

        if st.session_state.get("messages"):
            return None

        st.markdown(
            """
            <div class="welcome-box">

            <h3>👋 Welcome!</h3>

            <p>
            Ask a question and the RAG system will retrieve
            relevant information from the document.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 💡 Try asking")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "What is RAG?",
                use_container_width=True
            ):
                return "What is RAG?"

        with col2:
            if st.button(
                "What are RAG applications?",
                use_container_width=True
            ):
                return "What are RAG applications?"

        with col3:
            if st.button(
                "What is Agentic RAG?",
                use_container_width=True
            ):
                return "What is Agentic RAG?"

        return None