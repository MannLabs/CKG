from ckg.graphdb_connector import connector
from ckg import ckg_utils
from ckg.graphdb_builder import builder_utils
import os.path
import time
from collections import defaultdict
import re
import gzip

try:
    ckg_config = ckg_utils.read_ckg_config()
    oconfig = builder_utils.setup_config('ontologies')
    dbconfig = builder_utils.setup_config('databases')
except Exception as err:
    raise Exception("mapping - Reading configuration > {}.".format(err))


def reset_mapping(entity):
    """
    Checks if mapping.tsv file exists and removes it.

    :param str entity: entity label as defined in databases_config.yml
    """
    if entity in dbconfig["sources"]:
        directory = os.path.join(ckg_config["databases_directory"], dbconfig["sources"][entity])
        mapping_file = os.path.join(directory, "complete_mapping.tsv")
        if os.path.exists(mapping_file):
            os.remove(mapping_file)



def mark_complete_mapping(entity):
    """
    Checks if mapping.tsv file exists and renames it to complete_mapping.tsv.

    :param str entity: entity label as defined in databases_config.yml
    """
    if entity in dbconfig["sources"]:
        directory = os.path.join(ckg_config["databases_directory"], dbconfig["sources"][entity])
        mapping_file = os.path.join(directory, "mapping.tsv")
        new_mapping_file = os.path.join(directory, "complete_mapping.tsv")
        if os.path.exists(mapping_file):
            os.rename(mapping_file, new_mapping_file)


def getMappingFromOntology(ontology, source=None):
    """
    Converts .tsv file with complete list of ontology identifiers and aliases,
    to dictionary with aliases as keys and ontology identifiers as values.

    :param str ontology: ontology label as defined in ontologies_config.yml.
    :param source: name of the source database for selecting aliases.
    :type source: str or None
    :return: Dictionary of aliases (keys) and ontology identifiers (values).
    """
    mapping = {}
    ont = oconfig["ontologies"][ontology]
    dirFile = os.path.join(ckg_config["ontologies_directory"], ont)
    mapping_file = os.path.join(dirFile, "complete_mapping.tsv")
    max_wait = 0
    while not os.path.isfile(mapping_file) and max_wait < 5000:
        time.sleep(5)
        max_wait += 1
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                data = line.rstrip("\r\n").split("\t")
                if data[1] == source or source is None:
                    mapping[data[2].lower()] = data[0]
    except Exception:
        raise Exception("mapping - No mapping file {} for entity {}".format(mapping_file, ontology))

    return mapping


def getMappingFromDatabase(id_list, node, attribute_from='id', attribute_to='name'):
    id_list = ["'{}'".format(i) for i in id_list]
    driver = connector.getGraphDatabaseConnectionConfiguration()
    mapping_query = "MATCH (n:{}) WHERE n.{} IN [{}] RETURN n.{} AS from, n.{} AS to"
    mapping = connector.getCursorData(driver, mapping_query.format(node, attribute_from, ','.join(id_list), attribute_from, attribute_to))
    if not mapping.empty:
        mapping = dict(zip(mapping['from'], mapping['to']))

    return mapping


def getMappingForEntity(entity):
    """
    Converts .tsv file with complete list of entity identifiers and aliases, \
    to dictionary with aliases as keys and entity identifiers as values.

    :param str entity: entity label as defined in databases_config.yml.
    :return: Dictionary of aliases (keys) and entity identifiers (value).
    """
    mapping = {}
    if entity in dbconfig["sources"]:
        mapping_file = os.path.join(ckg_config["databases_directory"], os.path.join(dbconfig["sources"][entity], "complete_mapping.tsv"))
        max_wait = 0
        while not os.path.isfile(mapping_file) and max_wait < 5000:
            time.sleep(15)
            max_wait += 1

        try:
            with open(mapping_file, 'r', encoding='utf-8') as mf:
                for line in mf:
                    data = line.rstrip("\r\n").split("\t")
                    if len(data) > 1:
                        ident = data[0]
                        alias = data[1]
                        mapping[alias] = ident
        except Exception as err:
            raise Exception("mapping - No mapping file {} for entity {}. Error: {}".format(mapping_file, entity, err))

    return mapping


def getMultipleMappingForEntity(entity):
    """
    Converts .tsv file with complete list of entity identifiers and aliases, \
    to dictionary with aliases to other databases as keys and entity identifiers as values.

    :param str entity: entity label as defined in databases_config.yml.
    :return: Dictionary of aliases (keys) and set of unique entity identifiers (values).
    """
    mapping = defaultdict(set)
    if entity in dbconfig["sources"]:
        mapping_file = os.path.join(ckg_config["databases_directory"], os.path.join(dbconfig["sources"][entity], "complete_mapping.tsv"))
        max_wait = 0
        while not os.path.isfile(mapping_file) and max_wait < 5000:
            time.sleep(5)
            max_wait += 1

        try:
            with open(mapping_file, 'r') as mf:
                for line in mf:
                    data = line.rstrip("\r\n").split("\t")
                    if len(data) > 1:
                        ident = data[0]
                        alias = data[1]
                        mapping[alias].add(ident)
        except Exception:
            raise Exception("mapping - No mapping file {} for entity {}".format(mapping, entity))

    return mapping


def get_STRING_mapping_url(db="STRING"):
    """
    Get the url for downloading the mapping file from either STRING or STITCH

    :param str db: Which database to get the url from: STRING or STITCH
    :return: url where to download the mapping file
    """
    url = None
    config = builder_utils.get_config(config_name="stringConfig.yml", data_type='databases')
    if db.upper() == "STRING":
        url = config['STRING_mapping_url']
    elif db.upper() == "STITCH":
        url = config['STITCH_mapping_url']

    return url


def getSTRINGMapping(source="BLAST_UniProt_AC", download=True, db="STRING"):
    """
    Parses database (db) and extracts relationships between identifiers to order databases (source).

    :param str url: link to download database raw file.
    :param str source: name of the source database for selecting aliases.
    :param bool download: wether to download the file or not.
    :param str db: name of the database to be parsed.
    :return: Dictionary of database identifers (keys) and set of unique aliases to other databases (values).
    """
    url = get_STRING_mapping_url(db=db)
    mapping = defaultdict(set)
    directory = os.path.join(ckg_config["databases_directory"], db)
    file_name = os.path.join(directory, url.split('/')[-1])
    builder_utils.checkDirectory(directory)
    if download:
        print("Downloading", url, directory)
        builder_utils.downloadDB(url, directory)

    f = os.path.join(directory, file_name)
    first = True
    with gzip.open(f, 'rb') as mf:
        for line in mf:
            if first:
                first = False
                continue
            data = line.decode('utf-8').rstrip("\r\n").split("\t")
            if db == "STRING":
                stringID = data[0]
                alias = data[1]
                sources = data[2].split(' ')
            else:
                stringID = data[0]
                alias = data[2]
                sources = data[3].split(' ')
                if not alias.startswith('DB'):
                    continue

            if source in sources:
                mapping[stringID].add(alias)

    return mapping


def buildMappingFromOBO(oboFile, ontology, outputDir):
    """
    Parses and extracts ontology idnetifiers, names and synonyms from raw file, and writes all the information \
    to a .tsv file.
    :param str oboFile: path to ontology raw file.
    :param str ontology: ontology database acronym as defined in ontologies_config.yml.
    """
    cmapping_file = os.path.join(outputDir, "complete_mapping.tsv")
    mapping_file = os.path.join(outputDir, "mapping.tsv")
    identifiers = defaultdict(list)
    re_synonyms = r'\"(.+)\"'

    if os.path.exists(cmapping_file):
        os.remove(cmapping_file)

    with open(oboFile, 'r') as f:
        for line in f:
            if line.startswith("id:"):
                ident = ":".join(line.rstrip("\r\n").split(":")[1:])
            elif line.startswith("name:"):
                name = "".join(line.rstrip("\r\n").split(':')[1:])
                identifiers[ident.strip()].append(("NAME", name.lstrip()))
            elif line.startswith("xref:"):
                source_ref = line.rstrip("\r\n").split(":")[1:]
                if len(source_ref) == 2:
                    identifiers[ident.strip()].append((source_ref[0].strip(), source_ref[1]))
            elif line.startswith("synonym:"):
                synonym_type = "".join(line.rstrip("\r\n").split(":")[1:])
                matches = re.search(re_synonyms, synonym_type)
                if matches:
                    identifiers[ident.strip()].append(("SYN", matches.group(1).lstrip()))
    with open(mapping_file, 'w') as out:
        for ident in identifiers:
            for source, ref in identifiers[ident]:
                out.write(ident+"\t"+source+"\t"+ref+"\n")

    os.rename(mapping_file, cmapping_file)


def map_experiment_files(project_id, datasetPath, mapping):
    files = builder_utils.listDirectoryFiles(datasetPath)

    for file in files:
        outputfile = os.path.join(datasetPath, file)
        data = builder_utils.readDataset(outputfile)
        data = map_experimental_data(data, mapping)
        builder_utils.export_contents(data, datasetPath, file)


def map_experimental_data(data, mapping):
    mapping_cols = {}
    regex = "({})".format("|".join([re.escape(k) for k in sorted(list(mapping.keys()), key=len, reverse=True)]))
    if not data.empty:
        for column in data.columns:
            ids = re.search(regex, column)
            if ids is not None:
                ids = ids.group(1)
                mapping_cols[column] = column.replace(ids, mapping[ids])
            else:
                continue
        data = data.rename(columns=mapping_cols)

    return data


def get_mapping_analytical_samples(project_id):
    from ckg.graphdb_connector import connector
    driver = connector.getGraphDatabaseConnectionConfiguration()

    mapping = {}
    query = "MATCH (p:Project)-[:HAS_ENROLLED]-(:Subject)-[:BELONGS_TO_SUBJECT]-()-[:SPLITTED_INTO]-(a:Analytical_sample) WHERE p.id='{}' RETURN a.external_id, a.id".format(project_id)
    mapping = connector.getCursorData(driver, query)
    if not mapping.empty:
        mapping = mapping.set_index("a.external_id").to_dict(orient='dict')["a.id"]

    return mapping


if __name__ == "__main__":
    pass
