from typing import Optional

import streamlit as st

from session_manager import SessionStateManager


class ChatView:

    def __init__(
        self,
        session_manager: SessionStateManager
    ):
        self.session_manager = session_manager

    def render_history(self) -> None:

        messages = self.session_manager.get_messages()

        for message in messages:

            with st.chat_message(message.role):

                st.markdown(
                    message.content
                )

    @staticmethod
    def render_input() -> Optional[str]:

        return st.chat_input(
            "Ask something about your document...",
            key="chat_input"
        )

    @staticmethod
    def show_user_message(
        question: str
    ) -> None:

        with st.chat_message("user"):

            st.markdown(
                question
            )

    @staticmethod
    def show_assistant_message(
        answer: str
    ) -> None:

        st.markdown(
            answer
        )