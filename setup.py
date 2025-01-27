import os

#Find root directory of project
root_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

sam_path = os.path.expanduser("~/sam2")

#Create a .env file in the root directory
with open(root_dir + ".env", "w") as f:
    f.write(f"PYTHONPATH={sam_path}\n")