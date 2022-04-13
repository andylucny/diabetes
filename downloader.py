import requests
import zipfile
import os

filename = 'inputs.zip'
url = 'http://www.agentspace.org/download/'+filename
r = requests.get(url, allow_redirects=True)
open(filename, 'wb').write(r.content)

with zipfile.ZipFile(filename,"r") as zf:
    zf.extractall()
    
try:
    os.makedirs('outputs')
except:
    pass
try:
    os.makedirs('binaries')
except:
    pass
try:
    os.makedirs('subcontours')
except:
    pass
try:
    os.makedirs('annotations')
except:
    pass
try:
    os.makedirs('summaries')
except:
    pass
