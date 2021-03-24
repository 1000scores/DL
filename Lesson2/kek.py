import os

for root, dirs, files in os.walk("universum-photos"):
    print(files)