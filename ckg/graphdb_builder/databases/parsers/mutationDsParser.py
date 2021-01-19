import os.path
import re
from ckg.graphdb_builder import builder_utils

############################
#    IntAct - MutationDs   #
############################
def parser(databases_directory, download=True):
    relationships = set()
    config = builder_utils.get_config(config_name="mutationDsConfig.yml", data_type='databases')
    header = config['header']
    output_file_name = "mutation_curated_affects_interaction_with.tsv"
    regex = r":(\w+)\("
    url = config['mutations_url']
    directory = os.path.join(databases_directory, "MutationDs")
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
                pvariant= '_'.join(data[1].split(':'))
                effect = data[5]
                organism = data[10]
                interaction = data[11]
                evidence = data[12]

                if organism.startswith("9606"):
                    matches = re.finditer(regex, interaction)
                    for matchNum, match in enumerate(matches, start=1):
                        interactor = match.group(1)
                        relationships.add((pvariant, interactor, "CURATED_AFFECTS_INTERACTION_WITH", effect, interaction, evidence, internal_id, "Intact-MutationDs"))

    builder_utils.remove_directory(directory)

    return (relationships, header, output_file_name)
