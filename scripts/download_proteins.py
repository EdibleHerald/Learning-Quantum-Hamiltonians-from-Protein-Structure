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
    
    if "proteins" not in os.listdir():
        os.mkdir("proteins")
        wipe_dir = False    
    os.chdir("proteins")
    
    if dir_name not in os.listdir():
        os.mkdir(dir_name)
        wipe_dir = False
    os.chdir(dir_name)
    
    # Wipe directory?
    if wipe_dir and len(os.listdir()) < 2:
        items = os.listdir()
        
        for file in items:
            os.remove(file)
    
    for id in url_list:
        full_url = base_url + id + ext
        try:
            urllib.request.urlretrieve(full_url,id.upper() + ext)
        except:
            print(f"Protein {id} using url {full_url} is invalid.")
            continue
    
    # Go back two levels
    os.chdir("../../")

# Defining training set, validation set, and testing set

# Documenting exact category separation. All contain the
# Ser-His-Asp motif, for GNN pretraining / CNN training

trypsin = [
    "1PQ5",
    "1PQ7",
    "1XVM",
    "1XVO",
    "1GDU",
    "1PQ8",
    "1FY5",
    "1FY4",
    "1GDN",
    "1PPZ",
    "6YIY",
    "5JBB",
    "2AGE",
    "2AGI",
    "4M7G",
    "5MOO",
    "5XW9",
    "7AHV",
    "2F9N",
    "5MNB"
]

chymotrypsinogen_A = [
    "1YPH",
    "4CHA",
    "6DI8",
    "1N8O",
    "6T89",
    "1P2M",
    "1YPG",
    "4HGC",
    "9ATK",
    "1C5V",
    "1GG6",
    "1GL1",
    "1YPE",
    "2A2X",
    "2P8O",
    "3BIU",
    "5CHA",
    "6Y9H",
    "7JRX",
    "9U8G"
]

subtilisin = [
    "1S01",
    "1AK9",
    "1AQN",
    "1AU9",
    "3UNX",
    "4C3U",
    "5WRC",
    "6K2X",
    "8RSF",
    "2HPZ",
    "2PQ2",
    "2PYZ",
    "2V8B",
    "2WUV",
    "5ROU",
    "5ROI",
    "5RP4",
    "6K2R",
    "6QF1",
    "6RVE"
]

# Validation set, for preventing overfitting during training

validation_trypsin = [
    "6SVI",
    "9I24",
    "2CMY",
    "2G8T",
    "3A7V",
    "4DOQ",
    "4I8K"
]

validation_chymotrypsinogen_A = [
    "3I77",
    "3QWC",
    "3SI3",
    "6I51",
    "6T7H",
    "9ATU",
    "1AD8"
]

validation_subtilisin = [
    "3Q5G",
    "4CFY",
    "4FON",
    "5ARC",
    "5ROG",
    "5ROH",
    "5RP5"
]

# Test proteins, for testing against trained CNN:
test_trypsin = [
    "5MNX",
    "5MO0",
    "5MON",
    "5MOR",
    "5XW9",
    "1W12",
    "4MNV"
]

test_chymotrypsinogen_A = [
    "6YZC",
    "7AC9",
    "7Q0X",
    "1C5N",
    "1T8M",
    "1QUR",
    "2ZI2"
]

test_subtilisin = [
    "5RP9",
    "6TXG",
    "6Y5S",
    "7NUZ",
    "8RSG",
    "9FU1",
    "1C3L"
]

training_protein_names = trypsin + chymotrypsinogen_A + subtilisin
testing_protein_names = test_trypsin + test_chymotrypsinogen_A + test_subtilisin
validation_protein_names = validation_trypsin + validation_chymotrypsinogen_A + validation_subtilisin

download_proteins("training_proteins",training_protein_names,True)
download_proteins("testing_proteins",testing_protein_names,True)
download_proteins("validation_proteins",validation_protein_names,True)