from src.database.user_db import delete, get_all, get_by_id

def get_all_users(user_id):
    user = get_all()

    if not user:
        return ValueError("User not found") 
    
    return get_all()

def delete_user(id):
    id_list = []
    
    if isinstance(id, (list, tuple)):
        for x in id:
            id_list.append(int(x))
    else:
        id_list.append(int(id))

    return delete(id_list)