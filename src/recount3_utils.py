import pandas as pd

def parse_attributes(df, id_col, attributes_col):
    """
    Parse attributes where pipe (|) separates key-value pairs and ;; separates keys from values
    
    Parameters:
    df (DataFrame): Input dataframe
    id_col (str): Name of the ID column
    attributes_col (str): Name of the column containing the attributes
    
    Returns:
    DataFrame: New dataframe with parsed attributes as columns
    """
    # Create a list to store the parsed records
    records = []
    
    # Iterate through each row
    for _, row in df.iterrows():
        # Start with the ID
        record = {id_col: row[id_col]}
        
        # Split the attributes string by | to get each key-value pair
        if pd.notna(row[attributes_col]):
            pairs = row[attributes_col].split('|')
            
            # Split each pair by ;; and add to record
            for pair in pairs:
                if ';;' in pair:
                    key, value = pair.split(';;')
                    record[key.strip()] = value.strip()
        
        records.append(record)
    
    # Convert records to DataFrame
    return pd.DataFrame(records)

def create_rse_manual(project, project_home, organism, annotation, type):
    """
    Download and create a data structure from recount3 project data.
    Python equivalent of recount3::create_rse_manual R function.
    
    Parameters:
    project (str): Project identifier (e.g. "SRP150872")
    project_home (str): Data source path (e.g. "data_sources/sra")
    organism (str): Organism name (e.g. "human")
    annotation (str): Genome annotation version (e.g. "gencode_v26")
    type (str): Data type (e.g. "gene")
    
    Returns:
    DataFrame: Combined data from recount3 with metadata as additional columns
    """
    # Base URL for recount3 data
    base_url = "https://duffel.rail.bio/recount3"
    
    # Construct the URL path
    url_path = f"{base_url}/{organism}/{project_home}"
    
    # Define file names based on type
    counts_file = f"gene_sums/{project[-2:]}/{project}/sra.gene_sums.{project}.G026.gz"
    metadata_file = f"metadata/{project[-2:]}/{project}/sra.sra.{project}.MD.gz"
    
    # Download files
    counts_df = download_and_read_tsv(f"{url_path}/{counts_file}")
    metadata_df = download_and_read_tsv(f"{url_path}/{metadata_file}")
    
    # Parse metadata attributes
    if 'sample_attributes' in metadata_df.columns:
        sample_attributes = parse_attributes(
            metadata_df, 
            'external_id', 
            'sample_attributes'
        )
        
        # First merge metadata with sample attributes
        metadata_with_attrs = metadata_df.merge(
            sample_attributes,
            left_on='external_id',
            right_on='external_id',
            how='left'
        )
        
        # Melt the counts dataframe to long format
        # Keep 'gene_id' column and unpivot all SRR columns
        id_cols = ['gene_id']
        value_cols = [col for col in counts_df.columns if col.startswith('SRR')]
        
        counts_long = counts_df.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='external_id',
            value_name='counts'
        )
        
        # Now merge with metadata
        result = counts_long.merge(
            metadata_with_attrs,
            on='external_id',
            how='left'
        )
        
        return result
    
    return counts_df

def download_and_read_tsv(url):
    """
    Download and read a TSV file from a URL.
    
    Parameters:
    url (str): URL of the TSV file
    
    Returns:
    DataFrame: Parsed TSV data
    """
    import requests
    import io
    import gzip
    
    response = requests.get(url)
    response.raise_for_status()
        
    # Handle gzipped content
    content = gzip.decompress(response.content)
    return pd.read_csv(io.BytesIO(content), sep='\t', comment='#')


