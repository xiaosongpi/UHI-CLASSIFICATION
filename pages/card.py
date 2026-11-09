from src.services.user_service import delete_user
from src.services.auth_service import login, register, reset_password, update_password, create_session, delete_session, get_cookie
from src.database.client import get_supabase

import logging
import streamlit as st
import time

logger = logging.getLogger(__name__)
INTERNAL_ERROR = "Internal Server Error"

_INTERNAL_HINTS = (
    "nonetype", "object has no attribute", "object is not", "traceback",
    "keyerror", "typeerror", "attributeerror", "errno", "timed out",
    "connection", "http", "supabase", "postgrest", "jwt", "sqlstate",
    "gotrue", "response", "json", "unexpected",
)


def safe_message(error):
    """Tampilkan pesan ramah jika aman, selain itu cukup Internal Server Error."""
    message = str(error).strip()
    if not message or any(hint in message.lower() for hint in _INTERNAL_HINTS):
        logger.error("Error internal pada proses auth", exc_info=error)
        return INTERNAL_ERROR
    return message


def handle_recovery_code():
    code = st.query_params.get("code")
    if not code:
        return

    cookie = get_cookie()
    verifier = cookie.get("cv")

    if not verifier and not st.session_state.get("cv_wait"):
        st.session_state["cv_wait"] = True
        st.info("Memeriksa link reset password...")
        st.stop()

    st.session_state.pop("cv_wait", None)

    if not verifier:
        st.session_state["page"] = "login"
        st.session_state["auth_error"] = (
            "Verifier tidak ditemukan. Buka link di browser yang sama "
            "dengan yang dipakai meminta reset, lalu minta email baru."
        )
    else:
        try:
            sb = get_supabase()
            sb.auth._storage.set_item(f"{sb.auth._storage_key}-code-verifier", verifier)
            sb.auth.exchange_code_for_session({"auth_code": code})
            st.session_state["page"] = "new_password"
            cookie.remove("cv")
        except Exception as e:
            logger.error("Gagal memproses link reset password", exc_info=e)
            st.session_state["page"] = "login"
            st.session_state["auth_error"] = INTERNAL_ERROR

    st.session_state["open_dialog"] = True
    st.query_params.clear()
    st.rerun()

def go(page):
    st.session_state["page"] = page
    st.session_state["open_dialog"] = True
    st.rerun()

@st.dialog("Login")
def login_page():
    if err := st.session_state.pop("auth_error", None):
        st.error(err)
    email = st.text_input("Email", placeholder = "Masukkan email disini", type = "email", required = True, max_chars = 255)
    password = st.text_input("Password", placeholder = "Masukkan password disini", type = "password", required = True, icon = ":material/key:", max_chars = 255)

    if st.button("Login", type = "primary", width = "stretch"):
        try:
            with st.spinner("Login pengguna"):
                user = login(email, password)

            st.session_state["user"] = user 
            create_session(user)
            st.success("Login berhasil")
            time.sleep(1)
            st.rerun()

        except ValueError as e:
            st.error(safe_message(e))

        except Exception as e:
            logger.error("Error tak terduga", exc_info=e)
            st.error(INTERNAL_ERROR)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Register akun", type="tertiary", width = "stretch"):
            go("register")

    with col2:
        if st.button("Ganti password", type="tertiary", width = "stretch"):
            go("change_password")

@st.dialog("Register")
def register_page():
    with st.form("register_form", clear_on_submit=False, border = False):
        username = st.text_input("Username", placeholder="Masukkan username disini", icon=":material/person:", key="register_username", max_chars = 30)
        email = st.text_input("Email", placeholder="Masukkan email disini", icon=":material/mail:", key="register_email", max_chars = 255)
        password = st.text_input("Password", placeholder="Masukkan password disini", type="password", icon=":material/key:", key="register_password", max_chars = 255)
        password_confirm = st.text_input("Konfirmasi password", placeholder="Masukkan konfirmasi password disini", type="password", icon=":material/key:", key="register_password_confirm", max_chars = 255)

        submit = st.form_submit_button(
            "Register",
            type="primary",
            width="stretch"
        )

    if submit:
        if not password or not password_confirm:
            st.error("Password dan konfirmasi password wajib diisi.")

        else:
            try:
                with st.spinner("Registrasi pengguna baru"):
                    register(
                        username,
                        email,
                        password,
                        password_confirm
                    )

                st.success("Registrasi berhasil")
                go("login")

            except ValueError as e:
                st.error(safe_message(e))

            except Exception as e:
                logger.error("Error tak terduga", exc_info=e)
                st.error(INTERNAL_ERROR)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sudah punya akun", type="tertiary", width="stretch"):
            go("login")

    with col2:
        if st.button("Ganti password", type="tertiary", width="stretch"):
            go("change_password")

@st.dialog("Ganti Password")
def forget_password():
    email = st.text_input("Email", placeholder = "Masukkan email disini", type = "email", required = True, max_chars = 255)

    if st.button("Kirim email", type = "primary", width = "stretch"):
        try:
            with st.spinner("Mengirimkan email"):
                reset_password(email)
            
            st.success("Email telah terkirim")
        except ValueError as e:
            st.error(safe_message(e))
        except Exception as e:
            logger.error("Error tak terduga", exc_info=e)
            st.error(INTERNAL_ERROR)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sudah punya akun", type="tertiary", width = "stretch"):
            go("login")

    with col2:
        if st.button("Register akun", type="tertiary", width = "stretch"):
            go("register")

@st.dialog("Password Baru")
def new_password_page():
    password = st.text_input("Password", placeholder = "Masukkan password disini", type = "password", required = True, icon = ":material/key:", max_chars = 255)
    password_confirm = st.text_input("Komfirmasi password", placeholder = "Masukkan konfirmasi password disini", type = "password", required = True, icon = ":material/key:", max_chars = 255)

    if st.button("Simpan Password", type="primary", width="stretch"):
        try:
            with st.spinner("Mengirimkan email"):
                update_password(password, password_confirm)
                get_supabase().auth.sign_out()

            st.session_state["page"] = "login"
            st.session_state["open_dialog"] = True
            st.rerun()

        except ValueError as e:
            st.error(safe_message(e))

        except Exception as e:
            logger.error("Error tak terduga", exc_info=e)
            st.error(INTERNAL_ERROR)

def show_auth_dialog():
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    page = st.session_state["page"]

    if page == "login":
        login_page()
    elif page == "register":
        register_page()
    elif page == "change_password":
        forget_password()
    elif page == "new_password":
        new_password_page()

@st.dialog("Yakin ingin menghapus akun?")
def delete_account():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Tidak", type = "primary", width = "stretch"):
            st.rerun()
    with col2:
        if st.button("Iya", width = "stretch"):
            user_id = st.session_state["user"]["id"]
            delete_user(user_id)
            delete_session()
            st.session_state.clear()
            st.rerun()

def logout():
    delete_session()
    st.session_state.clear()
    time.sleep(1)
    st.rerun()