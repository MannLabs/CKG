import os.path
from collections import defaultdict
from lxml import etree
import zipfile
from ckg.graphdb_builder import mapping as mp, builder_utils

#################################
#   Human Metabolome Database   #
#################################
def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="hmdbConfig.yml", data_type='databases')
    directory = os.path.join(databases_directory, "HMDB")
    builder_utils.checkDirectory(directory)
    metabolites = extract_metabolites(config, directory, download)
    mapping = mp.getMappingFromOntology(ontology="Disease", source=config['HMDB_DO_source'])
    mapping.update(mp.getMappingFromOntology(ontology="Tissue", source=None))
    entities, attributes = build_metabolite_entity(config, directory, metabolites)
    relationships = build_relationships_from_HMDB(config, metabolites, mapping)
    entities_header = ['ID'] + attributes
    relationships_header = config['relationships_header']

    #builder_utils.remove_directory(directory)

    return (entities, relationships, entities_header, relationships_header)


def extract_metabolites(config, directory, download=True):
    metabolites = defaultdict()
    prefix = "{http://www.hmdb.ca}"
    url = config['HMDB_url']
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)
    fields = config['HMDB_fields']
    parentFields = config['HMDB_parentFields']
    structuredFields = config['HMDB_structures']
    with zipfile.ZipFile(fileName, 'r') as zipped:
        for zfile in zipped.namelist():
            zipped.extract(member=zfile, path=directory)
            xfile = os.path.join(directory, zfile)
            with open(xfile, 'rb') as f:
                context = etree.iterparse(f, events=("end",), tag=prefix + "metabolite")
                for _, elem in context:
                    values = {child.tag.replace(prefix, ''): child.text for child in elem.iterchildren() if child.tag.replace(prefix,'') in fields and child.text is not None}
                    for child in elem.iterchildren():
                        if child.tag.replace(prefix, '') in parentFields:
                            label = child.tag.replace(prefix, '')
                            values[label] = set()
                            for intchild in child.iter():
                                if intchild.text is not None:
                                    text = intchild.text
                                    if text.strip() != "":
                                        if label in structuredFields:
                                            if intchild.tag.replace(prefix, '') in structuredFields[label]:
                                                if len(structuredFields[label]) > 1:
                                                    values[intchild.tag.replace(prefix, '')] = text
                                                else:
                                                    values[label].add(text)
                                        elif intchild.tag.replace(prefix, '') in fields and text:
                                            values[label].add(text)
                    if "accession" in values:
                        metabolites[values["accession"]] = values

    return metabolites


def build_metabolite_entity(config, directory, metabolites):
    entities = set()
    attributes = config['HMDB_attributes']
    for metid in metabolites:
        entity = []
        entity.append(metid)
        for attr in attributes:
            if attr in metabolites[metid]:
                if type(metabolites[metid][attr]) == set:
                    lattr = ";".join(list(metabolites[metid][attr]))
                    entity.append(lattr)
                else:
                    entity.append(metabolites[metid][attr])
            else:
                entity.append('')
        entities.add(tuple(entity))

    build_HMDB_dictionary(directory, metabolites)

    return entities, attributes


def build_relationships_from_HMDB(config, metabolites, mapping):
    relationships = defaultdict(list)
    associations = config['HMDB_associations']
    for metid in metabolites:
        for ass in associations:
            ident = ass
            if len(associations[ass]) > 1:
                ident = associations[ass][1]
            if ass in metabolites[metid]:
                if type(metabolites[metid][ass]) == set:
                    for partner in metabolites[metid][ass]:
                        if partner.lower() in mapping:
                            partner = mapping[partner.lower()]
                        relationships[ident].append((metid, partner, associations[ass][0], "HMDB"))
                else:
                    partner = metabolites[metid][ass]
                    if metabolites[metid][ass].lower() in mapping:
                        partner = mapping[metabolites[metid][ass].lower()]
                    relationships[ident].append((metid, partner, associations[ass][0], "HMDB"))

    return relationships


def build_HMDB_dictionary(directory, metabolites):
    filename = "mapping.tsv"
    outputfile = os.path.join(directory, filename)
    mp.reset_mapping(entity="Metabolite")
    with open(outputfile, 'w', encoding='utf-8') as out:
        for metid in metabolites:
            if "name" in metabolites[metid]:
                name = metabolites[metid]["name"]
                out.write(metid+"\t"+name.lower()+"\n")
            if "synonyms" in metabolites[metid]:
                for synonym in metabolites[metid]["synonyms"]:
                    out.write(metid+"\t"+synonym.lower()+"\n")
            if "chebi_id" in metabolites[metid]:
                chebi_id = metabolites[metid]["chebi_id"]
                out.write(metid+"\t"+chebi_id+"\n")

    mp.mark_complete_mapping(entity="Metabolite")
