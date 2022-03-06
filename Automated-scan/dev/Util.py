import json

def load_settings():
    settings = {}
    with open('./Automated-scan/settings.json') as json_file:
        settings = json.load(json_file)
    return settings