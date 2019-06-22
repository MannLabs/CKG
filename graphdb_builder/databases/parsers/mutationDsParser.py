import os.path
import re
from collections import defaultdict
import ckg_utils
from graphdb_builder import builder_utils

############################
#    IntAct - MutationDs   # 
############################
def parser(databases_directory, download = True):
    relationships = set()
    cwd = os.path.abspath(os.path.dirname(__file__))
    config = ckg_utils.get_configuration(os.path.join(cwd, '../config/mutationDsConfig.yml'))
    header = config['header']
    output_file_name = "mutation_curated_affects_interaction_with.tsv"
    regex = r":(\w+)\("
    url = config['mutations_url']
    directory = os.path.join(databases_directory,"MutationDs")
    builder_utils.checkDirectory(directory)
    file_name = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)

    with open(file_name, 'r') as mf:
        first = True
        for line in mf:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
            if len(data) > 12: 
                internal_id = data[0] 
                pvariant= data[1]
                effect = data[5]
                protein = data[7].split(':')
                organism = data[10]
                interaction = data[11]
                evidence = data[12]
                
                if organism.startswith("9606") and len(protein) > 1:
                    protein = protein[1]
                    pvariant = protein+"_"+pvariant
                    matches = re.finditer(regex, interaction)
                    for matchNum, match in enumerate(matches, start=1):
                        interactor = match.group(1)
                        relationships.add((pvariant, interactor, "CURATED_AFFECTS_INTERACTION_WITH", effect, interaction, evidence, internal_id, "Intact-MutationDs"))
    return (relationships, header, output_file_name)
