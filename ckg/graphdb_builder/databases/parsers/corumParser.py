import os.path
import zipfile
from ckg.graphdb_builder import builder_utils
from collections import defaultdict

###################
#       CORUM     #
###################
def parser(databases_directory, download=True):
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory, "CORUM")
    builder_utils.checkDirectory(directory)

    try:
        config = builder_utils.get_config(config_name="corumConfig.yml", data_type='databases')
    except Exception as err:
        raise Exception("Reading configuration > {}.".format(err))

    database_url = config['database_url']
    entities_header = config['entities_header']
    relationships_headers = config['relationships_headers']
    zipped_fileName = os.path.join(directory, database_url.split('/')[-1])
    fileName = '.'.join(database_url.split('/')[-1].split('.')[0:2])
    if download:
        builder_utils.downloadDB(database_url, directory)
    names = set()
    first = True
    with zipfile.ZipFile(zipped_fileName) as z:
        with z.open(fileName) as f:
            for line in f:
                if first:
                    first = False
                    continue
                data = line.decode("utf-8").rstrip("\r\n").split("\t")
                identifier = data[0]
                name = data[1]
                organism = data[2]
                synonyms = data[3].split(';') if data[3] != "None" else [""]
                cell_lines = data[4].join(';')
                subunits = data[5].split(';')
                evidences = data[7].split(';')
                processes = data[8].split(';')
                pubmedid = data[14]

                if organism == "Human":
                    #ID name organism synonyms source
                    if name not in names:
                        entities.add((identifier, name, "9606", ",".join(synonyms), "CORUM"))
                        names.add(name)
                    for subunit in subunits:
                        #START_ID END_ID type cell_lines evidences publication source
                        relationships[("Protein", "is_subunit_of")].add((subunit, identifier, "IS_SUBUNIT_OF", ",".join(cell_lines), ",".join(evidences), pubmedid, "CORUM"))
                    for process in processes:
                        #START_ID END_ID type evidence_type score source
                        relationships["Biological_process", "associated_with"].add((identifier, process, "ASSOCIATED_WITH", "CURATED", 5, "CORUM"))

    builder_utils.remove_directory(directory)

    return entities, relationships, entities_header, relationships_headers


if __name__ == "__main__":
    pass
