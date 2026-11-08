from src.database.client import get_supabase
from datetime import datetime, timezone

def get_all():
    return (
        get_supabase().table("users")
        .select("*")
        .order("role", desc = True)
        .order("created_at", desc = True)
        .execute()
        .data
    )

def get_by_id(id):
    return (
        get_supabase().table("users")
        .select("*")
        .eq("id", id)
        .maybe_single()
        .execute()
        .data
    )

def get_by_email(email):
    return (
        get_supabase().table("users")
        .select("*")
        .eq("email", email)
        .maybe_single()
        .execute()
        .data
    )

def get_by_auth_id(auth_id):
    return (
        get_supabase().table("users")
        .select("*")
        .eq("auth_id", auth_id)
        .maybe_single()
        .execute()
        .data
    )

def create(auth_id, username, email, role="user"):
    return (
        get_supabase().table("users")
        .insert({
            "auth_id": auth_id,
            "username": username,
            "email": email,
            "role": role
        })
        .execute()
        .data
    )

def delete(user_id):
    return (
        get_supabase()
        .table("users")
        .delete()
        .in_("id", user_id)
        .execute()
        .data
    )