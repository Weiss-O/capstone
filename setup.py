import os

#Find root directory of project
root_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

sam_path = os.path.expanduser("~/sam2")

#Create a .env file in the root directory if it doesn't exist
env_file = root_dir + ".env"
if not os.path.exists(env_file):
    with open(env_file, "w") as f:
        f.write(f"PYTHONPATH={sam_path}\n")