from gql import Client,gql
from gql.transport.requests import RequestsHTTPTransport

# Use RCSB Data API to fetch protein chains. Then use
# Sequence Coordinates Server API to fetch catalytic site residue numbers.
# (Using GraphQL)

def fetch_protein_chains(protein_name:str):
    # First, get all chains of a given protein:
    transport = RequestsHTTPTransport(
        url="https://data.rcsb.org/graphql",
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
    
    try:
        result = client.execute(query)
        
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
            raise Exception(f"Unable to find catalytic site values: chains for {protein_name} failed to be fetched.")
            
        return chain_list,auth_id_list
    except Exception as e:
        print(f"Exception occured: {e}")

def fetch_catalytic_site_values(protein_name:str):
    # Fetch protein chain names first
    chain_list,auth_id_list = fetch_protein_chains(protein_name=protein_name)
    
    # Setup GraphQL request
    transport = RequestsHTTPTransport(
        url="https://sequence-coordinates.rcsb.org/graphql",
        verify=True,
        retries=3
    )

    client = Client(transport=transport,fetch_schema_from_transport=True)

    # Set is used to hold results as two chains in the 
    # same entity can reference the same active site.
    site_values = dict()

    # Fetch all catalytic site numbers from all chains.
    for name,auth_id in zip(chain_list,auth_id_list):
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
        results = client.execute(query)
        
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
