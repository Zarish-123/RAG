import streamlit as st


class StyleManager:

    @staticmethod
    def apply() -> None:

        st.markdown(
            """
            <style>

            .main-title {
                font-size: 42px;
                font-weight: 700;
                margin-bottom: 5px;
            }

            .subtitle {
                font-size: 17px;
                color: #777;
                margin-bottom: 25px;
            }

            .welcome-box {
                padding: 25px;
                border-radius: 15px;
                background-color: #f5f7fb;
                margin-bottom: 20px;
            }

            /* Chat messages */
            [data-testid="stChatMessage"] p {
                font-size: 19px !important;
                line-height: 1.7 !important;
            }

            /* Chat input text */
            [data-testid="stChatInput"] textarea {
                font-size: 18px !important;
                min-height: 60px !important;
            }

            [data-testid="stChatInput"] textarea::placeholder {
                font-size: 17px !important;
            }

            /* Sidebar buttons */
            [data-testid="stSidebar"] .stButton button {
                font-size: 16px !important;
            }

            </style>
            """,
            unsafe_allow_html=True
        )