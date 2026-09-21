import streamlit as st

from api_client import FastApiClient
from config import AppConfig
from controller import ChatController
from session_manager import SessionStateManager
from styles import StyleManager

from views.chat_view import ChatView
from views.header_view import HeaderView
from views.sidebar_view import SidebarView
from views.welcome_view import WelcomeView


class RagStreamlitApp:

    def __init__(self):

        self.config = AppConfig()

        self.session_manager = (
            SessionStateManager()
        )

        self.api_client = FastApiClient(
            self.config.api_url
        )

        self.chat_view = ChatView(
            self.session_manager
        )

        self.controller = ChatController(
            api_client=self.api_client,
            session_manager=self.session_manager,
            chat_view=self.chat_view
        )

    def setup(self) -> None:

        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )

        self.session_manager.initialize()

        StyleManager.apply()

    def run(self) -> None:

        self.setup()

        # Sidebar
        top_k = SidebarView(
            self.config,
            self.session_manager
        ).render()

        # Header
        HeaderView.render()

        # Welcome section
        suggestion = WelcomeView.render()

        # Previous messages
        self.chat_view.render_history()

        # Always render the chat input
        question = self.chat_view.render_input()

        # If user clicked a suggested question,
        # use that question instead.
        if suggestion:
            question = suggestion

        # Process question
        if question:

            self.controller.process_question(
                question=question,
                top_k=top_k
            )


def main():

    app = RagStreamlitApp()

    app.run()


if __name__ == "__main__":
    main()