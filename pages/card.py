from src.services.user_service import delete_user
from src.services.auth_service import login, register, reset_password, update_password, create_session, delete_session
from src.database.client import get_supabase

import streamlit as st
import time

if "code" in st.query_params:
    code = st.query_params["code"]

    try:
        get_supabase().auth.exchange_code_for_session({
            "auth_code": code
        })

        st.session_state["page"] = "new_password"

    except Exception as e:
        st.session_state["page"] = "login"
        st.session_state["auth_error"] = str(e)

    st.session_state["open_dialog"] = True
    st.query_params.clear()
    st.rerun()

def go(page):
    st.session_state["page"] = page
    st.rerun()

@st.dialog("Login")
def login_page():
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
            st.error(str(e))

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
                st.error(str(e))

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
            st.rerun()

        except ValueError as e:
            st.error(str(e))

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
            
            st.success("Password telah terupdate")
            st.session_state["page"] = "login"
            st.rerun()

        except ValueError as e:
            st.error(str(e))

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