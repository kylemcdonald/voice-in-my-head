import requests
import json
import urllib

def room_url_to_name(room_url):
    return urllib.parse.urlparse(room_url).path[1:]

# only needs to happen once per domain
def enable_transcription(daily_api_key, deepgram_api_key):
    print("enabling transcription")
    url = "https://api.daily.co/v1/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {daily_api_key}",
    }
    data = {"properties": {"enable_transcription": f"deepgram:{deepgram_api_key}"}}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("transcription enabled")
    return response


# official https://github.com/daily-demos/llm-talk/blob/main/auth.py
# uses room-specific token and token_expiry
def get_meeting_token(daily_api_key):
    print("getting meeting token")
    url = "https://api.daily.co/v1/meeting-tokens"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {daily_api_key}"}
    data = {"properties": {"is_owner": True}}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("got meeting token")
    if response.status_code == 200:
        return response.json()["token"]
    else:
        return None

def list_rooms(daily_api_key):
    print("listing rooms")
    url = "https://api.daily.co/v1/rooms"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {daily_api_key}"}
    response = requests.get(url, headers=headers)
    print("listed rooms")
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def list_meetings(daily_api_key):
    print("listing meetings")
    url = "https://api.daily.co/v1/meetings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {daily_api_key}"}
    response = requests.get(url, headers=headers)
    print("listed meetings")
    if response.status_code == 200:
        return response.json()
    else:
        return None    
    
def delete_room(daily_api_key, room_name):
    print(f"deleting room {room_name}")
    url = f"https://api.daily.co/v1/rooms/{room_name}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {daily_api_key}"}
    response = requests.delete(url, headers=headers)
    print("deleted room")
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def delete_completed(daily_api_key):
    meetings = list_meetings(daily_api_key)
    deleted_rooms = []
    for meeting in meetings["data"]:
        if meeting["ongoing"]:
            continue
        room = meeting["room"]
        if delete_room(daily_api_key, room) is None:
            break
        deleted_rooms.append(room)
    return deleted_rooms