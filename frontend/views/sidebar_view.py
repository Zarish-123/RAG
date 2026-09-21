import streamlit as st

from config import AppConfig
from session_manager import SessionStateManager


class SidebarView:

    def __init__(
        self,
        config: AppConfig,
        session_manager: SessionStateManager
    ):

        self.config = config
        self.session_manager = session_manager

    def render(self) -> int:

        with st.sidebar:

            st.title("⚙️ Settings")

            # New Chat
            if st.button(
                "➕ New Chat",
                use_container_width=True
            ):

                self.session_manager.new_chat()

                st.rerun()

            st.markdown("---")

            # Previous Chats
            st.markdown("### 💬 Previous Chats")

            chat_names = (
                self.session_manager.get_chat_names()
            )

            active_chat = (
                self.session_manager.get_active_chat()
            )

            for index, chat_name in enumerate(
                chat_names
            ):

                if chat_name == active_chat:

                    button_label = f"🟢 {chat_name}"

                else:

                    button_label = f"💬 {chat_name}"

                if st.button(
                    button_label,
                    key=f"previous_chat_{index}",
                    use_container_width=True
                ):

                    self.session_manager.switch_chat(
                        chat_name
                    )

                    st.rerun()

            st.markdown("---")

            # Knowledge Base
            st.markdown("### 📚 Knowledge Base")

            st.info(
                f"📄 {self.config.document_name}"
            )

            st.markdown("---")

            # Top K
            top_k = st.slider(
                "Number of retrieved chunks",
                min_value=1,
                max_value=self.config.max_top_k,
                value=self.config.default_top_k
            )

            st.markdown("---")

            # Clear current chat
            if st.button(
                "🗑️ Clear Current Chat",
                use_container_width=True
            ):

                self.session_manager.clear()

                st.rerun()

            st.markdown("---")

            st.caption(
                "Powered by FastAPI + Ollama + FAISS"
            )

        return top_k