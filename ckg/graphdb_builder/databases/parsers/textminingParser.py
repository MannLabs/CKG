import os.path
import pandas as pd
from ckg.graphdb_builder import mapping as mp, builder_utils

#####################################
#   Textmining (JensenLab.org)      #
#####################################


def parser(databases_directory, importDirectory, download=True):
    config = builder_utils.get_config(config_name="jensenlabConfig.yml", data_type='databases')
    outputfileName = "Publications.tsv"
    url = config['db_url']
    ifile = config['organisms_file']
    organisms = str(config['organisms'])
    directory = os.path.join(databases_directory, "Jensenlab")
    builder_utils.checkDirectory(os.path.join(directory, "textmining"))

    if download:
        builder_utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory, "textmining"))

    ifile = os.path.join(directory, os.path.join("textmining", ifile))
    valid_pubs = read_valid_pubs(organisms, ifile)
    entities, header = parse_PMC_list(config, os.path.join(directory, "textmining"), download=download, valid_pubs=valid_pubs)
    num_entities = len(entities)
    outputfile = os.path.join(importDirectory, outputfileName)
    builder_utils.write_entities(entities, header, outputfile)
    entities = None

    for qtype in config['db_mentions_types']:
        parse_mentions(config, directory, qtype, importDirectory, download)

    builder_utils.remove_directory(os.path.join(directory, "textmining"))

    return (num_entities, outputfile)


def read_valid_pubs(organisms, organisms_file):
    pubs = set()
    with open(organisms_file, 'r') as idbf:
        for line in idbf:
            data = line.rstrip('\r\n').split('\t')
            if str(data[0]) in organisms:
                pubs.update(set(data[1].split(" ")))
    return list(pubs)


def parse_PMC_list(config, directory, download=True, valid_pubs=None):
    url = config['PMC_db_url']
    plinkout = config['pubmed_linkout']
    entities = set()
    fileName = os.path.join(directory, url.split('/')[-1])

    if download:
        builder_utils.downloadDB(url, directory)

    entities = pd.read_csv(fileName, sep=',', dtype=str, compression='gzip', low_memory=False)
    entities = entities[config['PMC_fields']]
    entities = entities[entities.iloc[:, 0].notnull()]
    entities = entities.set_index(list(entities.columns)[0])
    if valid_pubs is not None:
        valid_pubs = set(entities.index).intersection(valid_pubs)
        entities = entities.loc[list(valid_pubs)]

    entities['linkout'] = [plinkout.replace("PUBMEDID", str(int(pubmedid))) for pubmedid in list(entities.index)]
    entities.index.names = ['ID']
    entities['TYPE'] = 'Publication'
    entities = entities.reset_index()
    header = [c.replace(' ', '_').lower() if c not in ['ID', 'TYPE'] else c for c in list(entities.columns)]
    entities = entities.replace('\\\\', '', regex=True)
    entities = list(entities.itertuples(index=False))

    return entities, header


def parse_mentions(config, directory, qtype, importDirectory, download=True):
    url = config['db_url']
    ifile = config['db_mentions_files'][qtype]
    if qtype == "9606":
        mapping = mp.getSTRINGMapping(download=download)
    elif qtype == "-1":
        mapping = mp.getSTRINGMapping(source=config['db_sources']["Drug"], download=download, db="STITCH")

    filters = []
    if qtype in config['db_mentions_filters']:
        filters = config['db_mentions_filters'][qtype]
    entity1, entity2 = config['db_mentions_types'][qtype]
    outputfile = os.path.join(importDirectory, entity1 + "_" + entity2 + "_mentioned_in_publication.tsv")

    if download:
        builder_utils.downloadDB(url.replace("FILE", ifile), os.path.join(directory, "textmining"))
    ifile = os.path.join(directory, os.path.join("textmining", ifile))
    with open(outputfile, 'w') as f:
        f.write("START_ID\tEND_ID\tTYPE\n")
        with open(ifile, 'r') as idbf:
            for line in idbf:
                data = line.rstrip("\r\n").split('\t')
                id1 = data[0]
                pubmedids = data[1].split(" ")
                ident = []
                if qtype == "9606":
                    id1 = "9606."+id1
                    if id1 in mapping:
                        ident = mapping[id1]
                elif qtype == "-1":
                    if id1 in mapping:
                        ident = mapping[id1]
                elif qtype == "-26":
                    if id1.startswith("DOID"):
                        ident = [id1]
                else:
                    ident = [id1]

                for i in ident:
                    if i not in filters:
                        aux = pd.DataFrame(data={"Pubmedids": list(pubmedids)})
                        aux["START_ID"] = i
                        aux["TYPE"] = "MENTIONED_IN_PUBLICATION"
                        aux.to_csv(path_or_buf=f, sep='\t', header=False, index=False, quotechar='"', line_terminator='\n', escapechar='\\')
                        aux = None
