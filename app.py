import streamlit as st
from pages.card import show_auth_dialog, delete_account, logout, handle_recovery_code
from src.services.auth_service import restore_session
from streamlit_js_eval import streamlit_js_eval
from streamlit_cookies_controller import CookieController
import time
from src.services.auth_service import get_cookie

cookie = get_cookie()

st.set_option("client.showErrorDetails", "none")

st.session_state["_cookie"] = CookieController()

handle_recovery_code()
if st.session_state.get("page") != "new_password":
    restore_session()

RESTRICTED_WITHOUT_LOGIN = {
    "peta_uhi",
    "klasifikasi_uhi",
    "riwayat"
}

st.set_page_config(
    page_title = "JakUHI: Beranda",
    page_icon = "images/Logo.png",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.logo(
    "images/Logo.png",
    size = "large",
)

st.markdown(
    """
        <style>
            [data-testid="stMain"] [data-testid="stMainBlockContainer"] {
                display: flex;
                flex-direction: column;
                width: 100%;
                max-width: 1200px;
            }
            [data-testid="stSidebarHeader"] {
                position: relative;
            }
            [data-testid="stSidebarHeader"]::after {
                content: "JakUHI";
                position: absolute;
                left: 45px;
                top: 60%;
                transform: translateY(-50%);
                font-size: 20px;
                font-weight: 700;
                white-space: nowrap;
                pointer-events: none;
            }
        </style>
    """, 
    unsafe_allow_html = True,
)

pages = [
    st.Page(
        "pages/home.py", 
        title="Beranda", 
        icon=":material/home:", 
        url_path = "beranda",
        default=True
    ),
    st.Page(
        "pages/uhi_classification.py", 
        title = "Klasifikasi UHI ", 
        icon = ":material/label:",
        url_path = "klasifikasi_uhi"
    ),
    st.Page(
        "pages/uhi_mapping.py", 
        title="Peta UHI", 
        icon=":material/map:",
        url_path = "peta_uhi"
    ),
    st.Page(
        "pages/history.py", 
        title = "Riwayat", 
        icon = ":material/history:",
        url_path = "riwayat"
    ),
    st.Page(
        "pages/performance.py", 
        title="Evaluasi Model", 
        icon=":material/trending_up:",
        url_path = "performa_model"
    ),
    st.Page(
        "pages/help.py", 
        title = "Bantuan", 
        icon = ":material/help:",
        url_path = "bantuan"
    ),
]

handle_recovery_code()
if st.session_state.get("page") != "new_password":
    restore_session()

user = st.session_state.get("user")
if not user and not st.session_state.get("cookie_checked"):
    st.session_state["cookie_checked"] = True
    time.sleep(0.5)
    st.rerun()

if user and user["role"] == "admin":
    pages.append(st.Page("pages/admin.py", title = "Pengguna", icon = ":material/person:", url_path = "user"))

page = st.navigation(
    pages = pages,
    position = "sidebar",
    expanded = True,
)

with st.sidebar:
    with st.container(key = "profile"):
        if user:
            st.markdown(f"##### Selamat datang, {user['username']}")
            with st.popover("Opsi"):
                if st.button("Logout", width = "stretch"):
                    logout()
                if st.button("Hapus akun", width = "stretch", type = "primary"):
                    delete_account()

        else:
            st.markdown("##### Not logged in")
            if st.button("Login", width = "stretch", type = "primary"):
                show_auth_dialog()        


dialog_opened = False
if st.session_state.pop("open_dialog", False):
    show_auth_dialog()
    dialog_opened = True

if page.url_path in RESTRICTED_WITHOUT_LOGIN and not user:
    if not dialog_opened:
        show_auth_dialog()
    st.stop()

# open_requested = st.session_state.pop("open_dialog", False)

# recovery = streamlit_js_eval(
#     js_expressions="""
#     (() => {
#         const params = new URLSearchParams(
#             window.parent.location.hash.substring(1)
#         );

#         return {
#             access_token: params.get("access_token"),
#             refresh_token: params.get("refresh_token"),
#             type: params.get("type")
#         };
#     })()
#     """,
#     key="recovery_token",
#     want_output=True
# )

# if recovery and recovery.get("type") == "recovery":
#     if not st.session_state.get("recovery_processed", False):
#         try:
#             get_supabase().auth.set_session(
#                 recovery["access_token"],
#                 recovery["refresh_token"]
#             )

#             st.session_state["recovery_processed"] = True
#             st.session_state["page"] = "new_password"
#             st.session_state["open_dialog"] = True

#             # Bersihkan token dari URL browser
#             streamlit_js_eval(
#                 js_expressions="""
#                 window.history.replaceState(
#                     null,
#                     '',
#                     window.location.pathname + window.location.search
#                 );
#                 true;
#                 """,
#                 key="clear_recovery_url",
#                 want_output=True
#             )

#             st.rerun()

#         except Exception as e:
#             st.error(f"Gagal memproses reset password: {e}")

page.run()
