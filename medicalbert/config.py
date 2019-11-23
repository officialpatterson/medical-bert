# We store all the application settings here.

# File locations
import json


def get_configuration(args):
    with open('default.json') as json_data:
        conf = json.load(json_data)

    for key, value in vars(args).items():
        if value:
            conf[key] = value
    return conf



