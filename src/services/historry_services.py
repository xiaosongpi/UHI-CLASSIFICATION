from src.database.history_db import get_by_user_id, create, delete

def get_history_by_user_id (id):
    return get_by_user_id(id)

def create_history (data):
    return create(data)

def delete_history(id):
    return delete(id)