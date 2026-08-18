import time,os,urllib
import urllib.request
# A temporary file to store urls and a function to download proteins from said urls

'''
download_proteins(dir_name:str,wipe_dir:bool=False)
dir_name: Name of directory where proteins are downloaded into.
url_list: Holds URLs as strings. Fetches from all URLs, ensure URL is safe/legitimate. 
wipe_dir: Wipe directory of all contents before downloading proteins into it.
'''
def download_proteins(dir_name:str,url_list:list(str),wipe_dir:bool=False) -> None:
    base_url = "https://files.rcsb.org/download/"
    
    # temporary
    ext = ".pdb"        
    
    if dir_name not in os.listdir():
        os.mkdir(dir_name)
        wipe_dir = False    
    os.chdir(dir_name)
    
    # Wipe directory?
    if wipe_dir:
        items = os.listdir()
        for file in items:
            os.remove(file)
        
    for id in url_list:
        full_url = base_url + id + ext
        urllib.request.urlretrieve(full_url,id.upper() + ext)
    pass

training_protein_urls = [
    "1A7R",
    "1ACX",
    "1BVL",
    "1DMC",
    "1ENH",
    "1I3V",
    "1P7I",
    "3A02",
    "3IY6",
    "3K2A",
    "4U7S",
    "4UT7",
    "5YD5",
    "5Z2S",
    "6J60",
    "6M91",
    "7F07",
    "9N97",
    "9CN2",
    "8SOW",
    "8SOZ",
    "6VUO",
    "7KBP",
    "6VRP",
    "1SVZ",
    "1MOE",
    "6OL7",
    "5C6W"
]

download_proteins("proteins",training_protein_urls,True)

