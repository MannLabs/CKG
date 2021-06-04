import os.path
from collections import defaultdict
import pandas as pd
from ckg.graphdb_builder import builder_utils
from ckg.graphdb_builder import mapping as mp


#########################
#         Pfam          #
#########################

def parser(databases_directory, import_directory, download=True, updated_on=None):
    config = builder_utils.get_config(config_name="pfamConfig.yml", data_type='databases')
    entity_header = config['entity_header']
    relationship_headers = config['relationship_headers']

    directory = os.path.join(databases_directory, 'Pfam')
    builder_utils.checkDirectory(directory)
    protein_mapping = mp.getMappingForEntity(entity="Protein")
    valid_proteins = list(set(protein_mapping.values()))

    ftp_url = config['ftp_url']
    filename = config['full_uniprot_file']
    # url = config['test']

    if not os.path.exists(os.path.join(directory, filename)):
        if download:
            builder_utils.downloadDB(ftp_url+filename, directory)

    stats = set()
    if os.path.exists(os.path.join(directory, filename)):
        fhandler = builder_utils.read_gzipped_file(os.path.join(directory, filename))
        identifier = None
        description = []
        lines = []
        missed = 0
        entities = set()
        relationships = defaultdict(set)
        is_first = True
        i = 0
        read_lines = 0
        num_entities = 0
        num_relationships = {}
        try:
            for line in fhandler:
                i += 1
                read_lines += 1
                if line.startswith("# STOCKHOLM"):
                    if identifier is not None:
                        entities.add((identifier, 'Functional_region', name, " ".join(description), "PFam"))
                        if len(entities) == 100:
                            print_files(entities, entity_header, outputfile=os.path.join(import_directory, 'Functional_region.tsv'), is_first=is_first)
                            num_entities += len(entities)
                            if 'mentioned_in_publication' in relationships:
                                print_files(relationships['mentioned_in_publication'], relationship_headers['mentioned_in_publication'], outputfile=os.path.join(import_directory, 'Functional_region_mentioned_in_publication.tsv'), is_first=is_first)
                                if 'mentioned_in_publication' not in num_relationships:
                                    num_relationships['mentioned_in_publication'] = 0
                                num_relationships['mentioned_in_publication'] += len(relationships['mentioned_in_publication'])
                            if 'found_in_protein' in relationships:
                                print_files(relationships['found_in_protein'], relationship_headers['found_in_protein'], outputfile=os.path.join(import_directory, 'Functional_region_found_in_protein.tsv'), is_first=is_first, filter_for=('END_ID', valid_proteins))
                                if 'found_in_protein' not in num_relationships:
                                    num_relationships['found_in_protein'] = 0
                                num_relationships['found_in_protein'] += len(relationships['found_in_protein'])
                            entities = set()
                            relationships = defaultdict(set)
                            is_first = False
                        identifier = None
                        description = []
                elif line.startswith("#=GF"):
                    data = line.rstrip('\r\n').split()
                    if 'AC' in data:
                        identifier = data[2].split('.')[0]
                    elif 'DE' in data:
                        name = " ".join(data[2:])
                    elif 'RM' in data:
                        relationships['mentioned_in_publication'].add((identifier, data[2], "MENTIONED_IN_PUBLICATION", "PFam"))
                    elif 'CC' in data:
                        description.append(" ".join(data[2:]))
                elif not line.startswith('//'):
                    data = line.rstrip('\r\n').split()
                    protein, positions = data[0].split('/')
                    protein = protein.replace('.', '-')
                    start, end = positions.split('-')
                    sequence = data[1]
                    relationships['found_in_protein'].add((identifier, protein, "FOUND_IN_PROTEIN", start, end, sequence, "PFam"))
                    if protein.split('-')[0] != protein:
                        relationships['found_in_protein'].add((identifier, protein.split('-')[0], "FOUND_IN_PROTEIN", start, end, sequence, "PFam"))
        except UnicodeDecodeError:
            lines.append(i)
            missed += 1

        fhandler.close()

        if len(entities) > 0:
            print_files(entities, entity_header, outputfile=os.path.join(import_directory,'Functional_region.tsv'), is_first=is_first)
            num_entities += len(entities)
            print_files(relationships['mentioned_in_publication'], relationship_headers['mentioned_in_publication'], outputfile=os.path.join(import_directory,'Functional_region_mentioned_in_publication.tsv'), is_first=is_first)
            num_relationships['mentioned_in_publication'] += len(relationships['mentioned_in_publication'])
            print_files(relationships['found_in_protein'], relationship_headers['found_in_protein'], outputfile=os.path.join(import_directory,'Functional_region_found_in_protein.tsv'), is_first=is_first)
            num_relationships['found_in_protein'] += len(relationships['found_in_protein'])

        stats.add(builder_utils.buildStats(num_entities, "entity", "Functional_region", "Pfam", 'Functional_region.tsv', updated_on))

        for rel in num_relationships:
            stats.add(builder_utils.buildStats(num_relationships[rel], "relationship", rel.upper(), "Pfam", 'Functional_region_'+rel+'.tsv', updated_on))

    builder_utils.remove_directory(directory)

    return stats


def print_files(data, header, outputfile, is_first, filter_for=None):
    df = pd.DataFrame(list(data), columns=header)
    if filter_for is not None:
        df = df[df[filter_for[0]].isin(filter_for[1])]
    if not df.empty:
        with open(outputfile, 'a', encoding='utf-8') as f:
            df.to_csv(path_or_buf=f, sep='\t',
                    header=is_first, index=False, quotechar='"',
                    line_terminator='\n', escapechar='\\')


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base, "../../../../data/databases")
    import_path = os.path.join(base, "../../../../data/imports/databases")

    parser(databases_directory=db_path, import_directory=import_path, download=False)
