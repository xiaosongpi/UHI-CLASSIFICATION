from src.database.client import get_supabase
from datetime import datetime, timezone

def get_by_user_id(user_id):
    return (
        get_supabase().table("histories")
        .select("*")
        .eq("user_id", user_id)
        .is_("deleted_at", "null")
        .order("created_at", desc=True)
        .execute()
        .data
    )

def create(data):
    return (
        get_supabase().table("histories")
        .insert(data)
        .execute()
        .data
    )

def delete(history_id):
    return (
        get_supabase().table("histories")
        .update({"deleted_at": datetime.now(timezone.utc).isoformat()})
        .eq("id", history_id)
        .execute()
        .data
    )