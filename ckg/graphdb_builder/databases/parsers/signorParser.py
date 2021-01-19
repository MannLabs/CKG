import os.path
from collections import defaultdict
from ckg.graphdb_builder import builder_utils


def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="signorConfig.yml", data_type='databases')

    directory = os.path.join(databases_directory, "SIGNOR")
    builder_utils.checkDirectory(directory)

    url = config['url']
    modifications = config['modifications']
    amino_acids = config['amino_acids']
    accronyms = config['accronyms']
    entities_header = config['entities_header']
    relationships_headers = config['rel_headers']

    entities = set()
    relationships = defaultdict(set)

    filename = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)

    entities, relationships = parse_substrates(filename, modifications, accronyms, amino_acids)

    return entities, relationships, entities_header, relationships_headers


def parse_substrates(filename, modifications, accronyms, amino_acids):
    entities = set()
    relationships = defaultdict(set)
    first = True
    with open(filename, 'r', encoding="utf-8") as fhandler:
        for line in fhandler:
            if first:
                first = False
                continue

            data = line.rstrip("\r\n").split("\t")
            source = data[2]
            target = data[6]
            regulation = data[8]
            mechanism = data[9]
            residue_mod = data[10]
            seq_window = data[11]
            organism = data[12]
            pubmedid = data[21]
            if organism == "9606" and mechanism in modifications and residue_mod != '':
                if len(residue_mod) > 3:
                    residue = ''.join(residue_mod[0:3])
                    position = ''.join(residue_mod[3:])
                if residue in amino_acids:
                    residue = amino_acids[residue]
                    modification  = modifications[mechanism]
                    if mechanism in accronyms:
                        modified_protein_id = target+"_"+residue+position+"-"+accronyms[mechanism]
                        entities.add((modified_protein_id, "Modified_protein", target, seq_window, position, residue, "SIGNOR"))
                        relationships[("Protein", "has_modified_site")].add((target, modified_protein_id, "HAS_MODIFIED_SITE", "SIGNOR"))
                        relationships[("Peptide", "has_modified_site")].add((seq_window.upper(), modified_protein_id, "HAS_MODIFIED_SITE", "SIGNOR"))
                        relationships[("Modified_protein", "has_modification")].add((modified_protein_id, modification, "HAS_MODIFICATION", "SIGNOR"))
                        relationships[('Substrate', 'is_substrate_of')].add((modified_protein_id, source,"IS_SUBSTRATE_OF", regulation,"CURATED", 5, "SIGNOR"))
                        if pubmedid != '':
                            relationships['Modified_protein_Publication', 'mentioned_in_publication'].add((pubmedid, modified_protein_id, "MENTIONED_IN_PUBLICATION"))

    return entities, relationships


if __name__ == "__main__":
    pass
