def create_gene_to_protein_mapping():

    import os
    import requests
    import config
    import conversions
    config = config.read_config()

    ## Download Biogrid human gene interactions
    if not os.path.exists(config['PATH_BIOGRID'] + config['BIOGRID_GI']):

        # Download the zip with all the species, then extract and delete the others
        if not os.path.exists(config['PATH_BIOGRID'] + config['BIOGRID_ALL']):
            print("Downloading Biogrid gene interactions...")
            request_result = requests.get(config['URL_BIOGRID_ALL'])
            os.makedirs(os.path.dirname(config['PATH_BIOGRID']), exist_ok=True)
            open(config['BIOGRID_ALL'], "w").write(request_result.text)

        # Decompress the file
        from zipfile import ZipFile
        with ZipFile(config['PATH_BIOGRID'] + config['BIOGRID_ALL'], 'r') as zip:
            print("Extracting human gene interactions file...")
            zip.extract(config['BIOGRID_GI'], config['PATH_BIOGRID'])

    print("Biogrid gene interactions READY")

    # Create list of unique gene ids
    genes_to_proteins = {}
    file_biogrid_gi = open(config['PATH_BIOGRID'] + config['BIOGRID_GI'])
    file_biogrid_gi.readline()
    for line in file_biogrid_gi:
        columns = line.split()
        genes_to_proteins[columns[1]] = None

    # Convert gene to protein ids by batches
    file_biogrid_gene_to_protein = open(config['PATH_BIOGRID'] + config['BIOGRID_PPI'], "w")
    batch_size = int(config['ID_MAPPING_BATCH_SIZE'])
    start, end = 0, batch_size
    total = len(genes_to_proteins.keys())
    while True:
        end = min(end, total)
        print(f"  Converting genes {start} to {end}")
        mapping = conversions.map_ids(list(genes_to_proteins.keys())[start:end])
        for key, values in mapping.items():
            for value in values:
                file_biogrid_gene_to_protein.write(f"{key}\t{value}\n")

        if end == total:
            break
        start += batch_size
        end += batch_size
    
    file_biogrid_gene_to_protein.close()