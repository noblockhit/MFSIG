import requests
from .state import State

def send_text_to_whatsapp(message):
    if State.whatsapp_number == "" or State.whatsapp_api_key == "":
        return
    
    message = message.replace(" ", "+")
    message = message.replace("&", "")

    request_template = f"https://api.callmebot.com/whatsapp.php?phone={State.whatsapp_number}&text={message}&apikey={State.whatsapp_api_key}"

    return requests.get(request_template)
