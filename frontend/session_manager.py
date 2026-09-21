from typing import List

import streamlit as st

from models import ChatMessage


class SessionStateManager:

    CHATS_KEY = "chats"
    ACTIVE_CHAT_KEY = "active_chat"

    def initialize(self) -> None:

        if self.CHATS_KEY not in st.session_state:
            st.session_state[self.CHATS_KEY] = {
                "New Chat": []
            }

        if self.ACTIVE_CHAT_KEY not in st.session_state:
            st.session_state[self.ACTIVE_CHAT_KEY] = "New Chat"

    def get_active_chat(self) -> str:
        return st.session_state[self.ACTIVE_CHAT_KEY]

    def get_messages(self) -> List[ChatMessage]:

        active_chat = self.get_active_chat()

        return st.session_state[
            self.CHATS_KEY
        ].get(active_chat, [])

    def add_message(
        self,
        role: str,
        content: str
    ) -> None:

        active_chat = self.get_active_chat()

        st.session_state[
            self.CHATS_KEY
        ][active_chat].append(
            ChatMessage(
                role=role,
                content=content
            )
        )

    def set_chat_title(
        self,
        title: str
    ) -> None:

        active_chat = self.get_active_chat()

        st.session_state[
            self.CHATS_KEY
        ][title] = st.session_state[
            self.CHATS_KEY
        ].pop(active_chat)

        st.session_state[
            self.ACTIVE_CHAT_KEY
        ] = title

    def new_chat(self) -> None:

        chats = st.session_state[self.CHATS_KEY]

        chat_number = len(chats) + 1

        chat_name = f"New Chat {chat_number}"

        chats[chat_name] = []

        st.session_state[
            self.ACTIVE_CHAT_KEY
        ] = chat_name

    def switch_chat(
        self,
        chat_name: str
    ) -> None:

        if chat_name in st.session_state[self.CHATS_KEY]:

            st.session_state[
                self.ACTIVE_CHAT_KEY
            ] = chat_name

    def get_chat_names(self) -> List[str]:

        return list(
            st.session_state[
                self.CHATS_KEY
            ].keys()
        )

    def clear(self) -> None:

        active_chat = self.get_active_chat()

        st.session_state[
            self.CHATS_KEY
        ][active_chat] = []

    def delete_chat(
        self,
        chat_name: str
    ) -> None:

        chats = st.session_state[self.CHATS_KEY]

        if len(chats) <= 1:
            return

        if chat_name in chats:
            del chats[chat_name]

        remaining_chats = list(chats.keys())

        if (
            st.session_state[self.ACTIVE_CHAT_KEY]
            not in remaining_chats
        ):

            st.session_state[
                self.ACTIVE_CHAT_KEY
            ] = remaining_chats[0]