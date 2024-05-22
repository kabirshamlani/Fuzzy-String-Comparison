import os
currpath=os.getcwd()
print(currpath)
try:
    os.system("pip install --no-cache-dir -r requirement.txt")
    os.system("pip3 install --no-cache-dir -r requirement.txt")
except:
    print("Error")    
os.system("fastapi dev server.py")
