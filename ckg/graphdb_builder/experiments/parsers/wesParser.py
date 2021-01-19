import os
import pandas as pd
import numpy as np
from ckg.graphdb_builder import builder_utils

def parser(projectId):
    data = {}
    config = builder_utils.get_config(config_name="wes.yml", data_type='experiments')
    directory = '../../../data/experiments/PROJECTID/wes/'
    if 'directory' in config:
        directory = config['directory']
    directory = directory.replace('PROJECTID', projectId)
    wes_data = parseWESDataset(projectId, config, directory)
    if wes_data is not None:
        somatic_mutations = pd.DataFrame()
        for sample in wes_data:
            entities, variantRows, sampleRows, geneRows, chrRows = extractWESRelationships(wes_data[sample], config)
            data[('somatic_mutation_known_variant', 'w')] = variantRows
            data[('somatic_mutation_sample', 'w')] = sampleRows
            data[('somatic_mutation_gene', 'w')] = geneRows
            data[('somatic_mutation_chromosome', 'w')] = chrRows
            if somatic_mutations.empty:
                somatic_mutations = entities
            else:
                new = set(entities.index).difference(set(somatic_mutations.index))
                somatic_mutations = somatic_mutations.append(entities.loc[new,:], ignore_index=False)
        somatic_mutations = somatic_mutations.reset_index()
        data[('somatic_mutation', 'w')] = somatic_mutations

    return data

def parseWESDataset(projectId, configuration, dataDir):
    datasets = {}
    files = builder_utils.listDirectoryFiles(dataDir)
    for dfile in files:
        filepath = os.path.join(dataDir, dfile)
        if os.path.isfile(filepath):
            sample, data = loadWESDataset(filepath, configuration)
            datasets[sample] = data

    return datasets

def loadWESDataset(uri, configuration):
    ''' This function gets the molecular data from a Whole Exome Sequencing experiment.
        Input: uri of the processed file resulting from the WES analysis pipeline. The resulting
        Annovar annotated VCF file from Mutect (sampleID_mutect_annovar.vcf)
        Output: pandas DataFrame with the columns and filters defined in config.py '''
    aux = uri.split("/")[-1].split("_")
    sample = aux[0]
    #Get the columns from config
    columns = configuration["columns"]
    #Read the data from file
    data = builder_utils.readDataset(uri)
    if configuration['filter'] in data.columns:
        data = data.loc[data[configuration['filter']], :]
    data = data[columns]
    data["sample"] = aux[0]
    data["variant_calling_method"] = aux[1]
    data["annotated_with"] = aux[2].split('.')[0]
    data["alternative_names"] = data[configuration["alt_names"]]
    data = data.drop(configuration["alt_names"], axis = 1)
    data = data.iloc[1:]
    data = data.replace('.', np.nan)
    data["ID"] = data[configuration["id_fields"]].apply(lambda x: str(x[0])+":g."+str(x[1])+str(x[2])+'>'+str(x[3]), axis=1)
    data.columns = configuration['new_columns']
    return sample, data

def extractWESRelationships(data, configuration):
    entityAux = data.copy()
    entityAux = entityAux.set_index("ID")

    variantAux = data.copy()
    variantAux = variantAux.rename(index=str, columns={"ID": "START_ID"})
    variantAux["END_ID"] = variantAux["START_ID"]
    variantAux = variantAux[["START_ID", "END_ID"]]
    variantAux["TYPE"] = "IS_KNOWN_VARIANT"
    variantAux = variantAux.drop_duplicates()
    variantAux = variantAux.dropna(how="any")
    variantAux = variantAux[["START_ID", "END_ID", "TYPE"]]

    sampleAux = data.copy()
    sampleAux = sampleAux.rename(index=str, columns={"ID": "END_ID", "sample": "START_ID"})
    sampleAux["TYPE"] = "HAS_MUTATION"
    sampleAux = sampleAux[["START_ID", "END_ID", "TYPE"]]

    geneAux = data.copy()
    geneAux = geneAux.rename(index=str, columns={"ID": "START_ID", "gene": "END_ID"})
    geneAux["TYPE"] = "VARIANT_FOUND_IN_GENE"
    geneAux = geneAux[["START_ID", "END_ID", "TYPE"]]
    s = geneAux["END_ID"].str.split(';').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
    del geneAux["END_ID"]
    aux = s.to_frame("END_ID")
    geneAux = geneAux.join(aux)

    chrAux = data.copy()
    chrAux = chrAux.rename(index=str, columns={"ID": "START_ID", "chr": "END_ID"})
    chrAux["END_ID"] = chrAux["END_ID"].str.replace("chr",'')
    chrAux["TYPE"] = "VARIANT_FOUND_IN_CHROMOSOME"
    chrAux = chrAux[["START_ID", "END_ID", "TYPE"]]

    return entityAux, variantAux, sampleAux, geneAux, chrAux
