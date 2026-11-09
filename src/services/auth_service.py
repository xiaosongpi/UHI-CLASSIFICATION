import streamlit as st
from streamlit_cookies_controller import CookieController
from src.database.client import get_supabase
from src.database import user_db

def get_cookie():
    if "_cookie" not in st.session_state:
        st.session_state["_cookie"] = CookieController()

    return st.session_state["_cookie"]

def register(username, email, password, password_confirm, role="user"):
    if not username:
        raise ValueError("Username tidak boleh kosong")
    if not email:
        raise ValueError("Email tidak boleh kosong")
    if not password:
        raise ValueError("Password tidak boleh kosong")
    if not password_confirm:
        raise ValueError("Konfirmasi password tidak boleh kosong")

    if len(password) < 8:
        raise ValueError("Password minimal 8 karakter")
    if password != password_confirm:
        raise ValueError("Password tidak sama")

    try:
        response = get_supabase().auth.sign_up({
            "email": email,
            "password": password,
        })

        if not response.user:
            raise ValueError("Gagal register pengguna")

        user_db.create(
            auth_id=response.user.id,
            username=username,
            email=email,
            role="user"
        )

        return response.user

    except Exception as e:
        raise ValueError(str(e))

def create_session(user):
    cookie = get_cookie()

    try:
        cookie.set("access", user["access_token"], max_age=2592000)
        cookie.set("refresh", user["refresh_token"], max_age=2592000)
    except TypeError:
        pass

def delete_session():
    get_supabase().auth.sign_out()
    st.session_state.pop("user", None)

def restore_session():
    if st.session_state.get("user"):
        return

    cookie = get_cookie()

    try:
        access = cookie.get("access")
        refresh = cookie.get("refresh")
    except TypeError:
        return

    if not access or not refresh:
        return

    try:
        res = get_supabase().auth.set_session(access, refresh)
        profile = user_db.get_by_auth_id(res.user.id)

        if not profile:
            raise ValueError("Profil pengguna tidak ditemukan")
        user = {
            "id": profile["id"],
            "auth_id": res.user.id,
            "username": profile["username"],
            "email": res.user.email,
            "role": profile["role"],
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "expires_at": res.session.expires_at,
        }
        st.session_state["user"] = user
        create_session(user)
    except Exception:
        try:
            cookie.remove("access")
            cookie.remove("refresh")
        except Exception:
            pass

def login(email, password):
    if not email:
        raise ValueError("Email tidak boleh kosong")
    if not password:
        raise ValueError("Password tidak boleh kosong")
    
    try:
        response = get_supabase().auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        user = user_db.get_by_auth_id(response.user.id)

        if not user:
            raise ValueError("Profil pengguna tidak ditemukan")

        return {
            "id": user["id"],
            "auth_id": response.user.id,
            "username": user["username"],
            "email": response.user.email,
            "role": user["role"],
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "expires_at": response.session.expires_at
        }

    except Exception as e:
        raise ValueError(str(e))

def reset_password(email):
    if not email:
        raise ValueError("Email tidak boleh kosong")

    sb = get_supabase()

    sb.auth.reset_password_for_email(
        email,
        options={
            "redirect_to": "https://jakuhi.streamlit.app/"
        }
    )

    verifier = sb.auth._storage.get_item(
        f"{sb.auth._storage_key}-code-verifier"
    )

    if verifier:
        cookie = get_cookie()
        cookie.set("cv", verifier, max_age=3600, same_site="lax")
        
def update_password(password, password_confirm):
    if not password:
        raise ValueError("Password tidak boleh kosong")
    if not password_confirm:
        raise ValueError("Konfirmasi password tidak boleh kosong")

    if len(password) < 8:
        raise ValueError("Password minimal 8 karakter")

    if password != password_confirm:
        raise ValueError("Password tidak sama")

    try:
        response = get_supabase().auth.update_user({
            "password": password
        })

        return response.user

    except Exception as e:
        raise ValueError(str(e))
