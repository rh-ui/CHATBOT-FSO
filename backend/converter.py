import json

def extract_intents(data_list):
    intent_list = []
    for entry in data_list:
        intent = entry.get("intent")
        if intent:
            intent_list.append({"intent": intent})
    return intent_list

    # Save to new file
   
    

