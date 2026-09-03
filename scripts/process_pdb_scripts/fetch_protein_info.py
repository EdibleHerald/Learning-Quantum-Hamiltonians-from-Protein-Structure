from gql import Client,gql
from gql.transport.requests import RequestsHTTPTransport
from diskcache import Cache
from hash_for_cache import query_response_encoder
# Use RCSB Data API to fetch protein chains. Then use
# Sequence Coordinates Server API to fetch catalytic site residue numbers.
# (Using GraphQL)

def fetch_protein_chains(protein_name:str):
    URL = "https://data.rcsb.org/graphql"
    
    # Fetch cached query response if available
    with Cache("__scriptcache__") as cache:
        
        # Get hash
        query_hash = query_response_encoder(protein=protein_name,api_url=URL).hexdigest()
        
        if query_hash in cache:
            # Fetch cached query
            result = cache[query_hash]
        else:
            # Fetch results
    
            transport = RequestsHTTPTransport(
                url=URL,
                verify=True,
                retries=3
            )
            
            client = Client(transport=transport,fetch_schema_from_transport=True)

            query = gql( # Define GraphQL query
                f"""
                query FetchProteinChains {{
                    entry(entry_id:"{protein_name}"){{
                        assemblies {{
                            polymer_entity_instances {{
                                rcsb_id
                                rcsb_polymer_entity_instance_container_identifiers{{
                                    auth_asym_id
                                }}
                            }}
                        }}
                    }}
                }}
                """
            )
            
            # Try fetching results
            try:
                result = client.execute(query)
            except Exception as e:
                print(f"Failed to fetch protein chain data from '{URL}'!")
                raise Exception(f"Failed to fetch protein. Reason: {e}")
    
    try:
        assemblies = result['entry']['assemblies']

        chain_list = [] 
        auth_id_list = []
        
        # Iterate through all listed assemblies 
        for i in range(len(assemblies)):
            dict_list = assemblies[i]['polymer_entity_instances'] # List of dictionaries containing chains (e.g. [{'rcsb_id': '1YPH.A'}] )
    
            # Grab all strings from dictionaries
            for id_dict in dict_list:
                chain_list.append(id_dict['rcsb_id'])
                auth_id_list.append(id_dict['rcsb_polymer_entity_instance_container_identifiers']['auth_asym_id'])
        
        if not chain_list or len(chain_list) == 0:
            raise Exception(f"No catalytic sites for protein {protein_name} found!")
        
        # Cache results if some catalytic site data is returned
        with Cache("__scriptcache__") as cache:
            if query_hash not in cache:
                cache[query_hash] = result

        return chain_list,auth_id_list
    except Exception as e:
        print(f"Exception occured: {e}")

def fetch_catalytic_site_values(protein_name:str):
    # Fetch protein chain names first
    chain_list,auth_id_list = fetch_protein_chains(protein_name=protein_name)
    
    URL = "https://sequence-coordinates.rcsb.org/graphql"
    
    # Setup GraphQL request
    transport = RequestsHTTPTransport(
        url=URL,
        verify=True,
        retries=3
    )

    client = Client(transport=transport,fetch_schema_from_transport=True)

    # Set is used to hold results as two chains in the 
    # same entity can reference the same active site.
    site_values = dict()

    # Fetch all catalytic site numbers from all chains.
    for name,auth_id in zip(chain_list,auth_id_list):
        # Check for cached query response
        with Cache("__scriptcache__") as cache:
            # Get hash
            query_hash = query_response_encoder(protein=f"{name}.{auth_id}",api_url=URL).hexdigest()
            if query_hash in cache:
                results = cache[query_hash]
            else:
                # Create query and fetch results

                query = gql(
                    f"""
                    query PDBChainUniProtAnnotations {{
                        annotations(
                            reference:PDB_INSTANCE
                            sources:[UNIPROT,PDB_INSTANCE]
                            queryId:"{name}"
                            filters:[{{
                                field:TYPE
                                operation:EQUALS
                                values:"ACTIVE_SITE"
                            }}]
                        ){{
                            features{{
                                type
                                feature_positions {{
                                    beg_seq_id
                                }}
                            }}
                        }}
                    }}
                    """
                )
                try:
                    results = client.execute(query)
                except Exception as e:
                    print(f"Failed to fetch protein catalytic site data from '{URL}'!")
                    raise Exception(f"Failed to fetch protein. Reason: {e}")
                
                if len(results['annotations']) > 0 and len(results['annotations'][0]['features']) > 0: # If results actually returns catalytic site value(s)
                    cache[query_hash] = results
                    
        # Grab all returned catalytic site values if one was returned.
        if len(results['annotations']) > 0:
            iter_list = results['annotations'][0]['features']
            values_list = []
            for feature_dict in iter_list:
                # Define dictionary to hold residue site values for a specific chain
                small_list = feature_dict['feature_positions']
                for seq_dict in small_list:
                    values_list.append(seq_dict['beg_seq_id'])
            site_values[auth_id] = values_list 

        else:
            continue # No active sites found
    
    return site_values # Returns list of dictionaries, matching each chain to their active site residue id's