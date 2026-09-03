from json import dumps
from hashlib import md5

# Functions to convert a protein and given parameter(s) into hashes to be used to caching

def voxel_json_encoder(protein:str,grid_size:int):
        # Create dictionary item, then turn into bytecode to be hashed
        dict_item = {'protein':protein,'graph_size':grid_size}
        return md5(dumps(dict_item).encode('utf-8'))
    
def graph_json_encoder(protein:str,distance_threshold:float):
        # Create dictionary item, then turn into bytecode to be hashed
        dict_item = {'protein':protein,'distance_threshold':distance_threshold}
        return md5(dumps(dict_item).encode('utf-8'))

# Functions for turning an API query response into a hash

def query_response_encoder(protein:str,api_url:str):
        # Hold protein name and api used
        dict_item = {'protein':protein,'api_url':api_url}
        return md5(dumps(dict_item).encode('utf-8'))
