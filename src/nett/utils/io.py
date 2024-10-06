import os
import sys
import json

def write_to_file(file_path, d) -> bool:
    with open(file_path, "w") as file:
        file.write(json.dumps(d))
    return True

def mute() -> None:
    sys.stdout = open(os.devnull, "w")
