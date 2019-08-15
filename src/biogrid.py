import os
import requests
import conversions

def create_gene_to_protein_mapping(path_biogrid, filename_gi, url, filename_all, filename_entrez_to_uniprot, batch_size):

    ## Download Biogrid human gene interactions
    if not os.path.exists(path_biogrid + filename_gi):

        # Download the zip with all the species, then extract and delete the others
        if not os.path.exists(path_biogrid + filename_all):
            print("Downloading Biogrid gene interactions...")
            request_result = requests.get(url)
            os.makedirs(os.path.dirname(path_biogrid, exist_ok=True))
            open(f"{path_biogrid}{filename_all}", "w").write(request_result.text)

        # Decompress the file
        from zipfile import ZipFile
        with ZipFile(path_biogrid + filename_all, 'r') as zip:
            print("Extracting human gene interactions file...")
            zip.extract(filename_gi, path_biogrid)
        os.remove(f"{path_biogrid}{filename_all}")

    print("Biogrid gene interactions READY")

    # Create list of unique gene ids
    genes_to_proteins = {}
    file_biogrid_gi = open(path_biogrid + filename_gi)
    file_biogrid_gi.readline()
    for line in file_biogrid_gi:
        columns = line.split()
        genes_to_proteins[columns[1]] = None
    #os.remove(f"{path_biogrid}{filename_gi}")

    # Convert gene to protein ids by batches
    file_biogrid_gene_to_protein = open(path_biogrid + filename_entrez_to_uniprot, "w")
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

def read_gene_interactions(path_biogrid, filename_gi):
    file_biogrid_gi = open(path_biogrid + filename_gi, "r")
    biogrid_gi = {}
    file_biogrid_gi.readline()
    for line in file_biogrid_gi:
        fields = line.split("\t")
        if len(fields) < 2:
            print(line)
        from_gene, to_gene = fields[1], fields[2]
        if from_gene > to_gene:
            from_gene, to_gene = to_gene, from_gene
        if not from_gene in biogrid_gi.keys():
            biogrid_gi[from_gene] = {to_gene}
        else:
            biogrid_gi[from_gene].add(to_gene)
    for key in biogrid_gi.keys():
        biogrid_gi[key] = tuple(biogrid_gi[key])
    return biogrid_gi

def read_uniprot_to_entrez_mapping(path_biogrid, filename_entrez_to_uniprot):
    file_entrez_to_uniprot = open(path_biogrid + filename_entrez_to_uniprot, "r")
    protein_to_genes = {}
    for line in file_entrez_to_uniprot:
        fields = line.split("\t")
        gene, protein = fields[0].strip(), fields[1].strip()
        if not protein in protein_to_genes.keys():
            protein_to_genes[protein] = {gene}
        else:
            protein_to_genes[protein].add(gene)
    for key in protein_to_genes.keys():
        protein_to_genes[key] = tuple(protein_to_genes[key])
    return protein_to_genes

def read_entrez_to_uniprot_mapping(path_biogrid, filename_entrez_to_uniprot):
    file_entrez_to_uniprot = open(path_biogrid + filename_entrez_to_uniprot, "r")
    genes_to_proteins = {}
    for line in file_entrez_to_uniprot:
        fields = line.split("\t")
        gene, protein = fields[0].strip(), fields[1].strip()
        if not gene in genes_to_proteins.keys():
            genes_to_proteins[gene] = {protein}
        else:
            genes_to_proteins[gene].add(protein)
    for key in genes_to_proteins.keys():
        genes_to_proteins[key] = tuple(genes_to_proteins[key])
    return genes_to_proteins

def create_ppi_dictionary(gi, entrez_to_uniprot):
    result = {}
    for gene, interactors in gi.items():
        if gene in entrez_to_uniprot.keys():
            for interactor in interactors:
                if interactor in entrez_to_uniprot.keys():
                    for uniprot_acc_1 in entrez_to_uniprot[gene]:
                        for uniprot_acc_2 in entrez_to_uniprot[interactor]:
                            if uniprot_acc_1 == uniprot_acc_2:
                                continue
                            if uniprot_acc_1 < uniprot_acc_2:
                                if not uniprot_acc_1 in result.keys():
                                    result[uniprot_acc_1] = {uniprot_acc_2}
                                else:
                                    result[uniprot_acc_1].add(uniprot_acc_2)
                            else:
                                if not uniprot_acc_2 in result.keys():
                                    result[uniprot_acc_2] = {uniprot_acc_1}
                                else:
                                    result[uniprot_acc_2].add(uniprot_acc_1)

    # Convert set values to tuple values
    for key in result.keys():
        result[key] = tuple(result[key])
    return result