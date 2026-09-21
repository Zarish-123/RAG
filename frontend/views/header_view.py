import streamlit as st


class HeaderView:

    @staticmethod
    def render() -> None:

        st.markdown(
            '<div class="main-title">🤖 RAG Assistant</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Ask questions from your knowledge base'
            '</div>',
            unsafe_allow_html=True
        )