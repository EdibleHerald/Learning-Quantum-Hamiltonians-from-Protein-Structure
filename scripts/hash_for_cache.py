from json import dumps
from hashlib import md5

# Here are defined two functions to convert a protein list and given parameters into hashes to be used to caching

def voxel_json_encoder(protein_list:list(str),grid_size:int):
        # Create dictionary item, then turn into bytecode to be hashed
        dict_item = {'protein_list':sorted(protein_list),'graph_size':grid_size}
        return md5(dumps(dict_item).encode('utf-8'))
    
def graph_json_encoder(protein_list:list(str),distance_threshold:float):
        # Create dictionary item, then turn into bytecode to be hashed
        dict_item = {'protein_list':sorted(protein_list),'distance_threshold':distance_threshold}
        return md5(dumps(dict_item).encode('utf-8'))