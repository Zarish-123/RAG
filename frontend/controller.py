import streamlit as st

from interfaces import ApiClientInterface
from session_manager import SessionStateManager
from views.chat_view import ChatView


class ChatController:

    def __init__(
        self,
        api_client: ApiClientInterface,
        session_manager: SessionStateManager,
        chat_view: ChatView
    ):
        self.api_client = api_client
        self.session_manager = session_manager
        self.chat_view = chat_view

    def process_question(
        self,
        question: str,
        top_k: int
    ) -> None:

        # Check whether this is the first question
        messages = self.session_manager.get_messages()

        if len(messages) == 0:

            title = question.strip()

            # Keep title short
            if len(title) > 35:
                title = title[:35].strip() + "..."

            self.session_manager.set_chat_title(
                title
            )

        # Save user question
        self.session_manager.add_message(
            "user",
            question
        )

        # Show user question
        self.chat_view.show_user_message(
            question
        )

        # Get answer from backend
        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching knowledge base..."
            ):

                result = self.api_client.ask(
                    question,
                    top_k
                )

            if not result.success:

                answer = (
                    "❌ Unable to connect to the "
                    "RAG backend.\n\n"
                    f"Error: {result.error}"
                )

                st.error(answer)

                self.session_manager.add_message(
                    "assistant",
                    answer
                )

                return

            # Show answer
            self.chat_view.show_assistant_message(
                result.answer
            )

            # Save answer
            self.session_manager.add_message(
                "assistant",
                result.answer
            )

        # Refresh page so sidebar gets updated title
        st.rerun()