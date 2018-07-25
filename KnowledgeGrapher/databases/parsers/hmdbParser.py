import os.path
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import hmdbConfig as iconfig
from collections import defaultdict
from lxml import etree
import zipfile
from KnowledgeGrapher import utils


#################################
#   Human Metabolome Database   # 
#################################
def parser(download = False):
    metabolites = defaultdict()
    prefix = "{http://www.hmdb.ca}"
    url = iconfig.HMDB_url
    relationships = set()
    directory = os.path.join(dbconfig.databasesDir,"HMDB")
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        utils.downloadDB(url, "HMDB")
    fields = iconfig.HMDB_fields
    parentFields = iconfig.HMDB_parentFields
    structuredFields = iconfig.HMDB_structures
    with zipfile.ZipFile(fileName, 'r') as zipped:
        for f in zipped.namelist():
            data = zipped.read(f) 
            root = etree.fromstring(data)
            context = etree.iterwalk(root, events=("end",), tag=prefix+"metabolite")
            for _,elem in context:
                values = {child.tag.replace(prefix,''):child.text for child in elem.iterchildren() if child.tag.replace(prefix,'') in fields and child.text is not None}
                for child in elem.iterchildren(): 
                    if child.tag.replace(prefix,'') in parentFields:
                        label = child.tag.replace(prefix,'')
                        values[label] = set()
                        for intchild in child.iter():
                            if intchild.text is not None and intchild.text.strip() != "":
                                if label in structuredFields:
                                    if intchild.tag.replace(prefix,'') in structuredFields[label]:
                                        if len(structuredFields[label]) >1:
                                            values[intchild.tag.replace(prefix,'')] = intchild.text
                                        else:
                                            values[label].add(intchild.text) 
                                elif intchild.tag.replace(prefix,'') in fields and intchild.text:
                                    values[label].add(intchild.text) 
                            
                if "accession" in values:
                    metabolites[values["accession"]] = values
    return metabolites

def build_metabolite_entity(metabolites):
    entities = set()
    attributes = iconfig.HMDB_attributes
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
    
    return entities, attributes
    
def build_relationships_from_HMDB(metabolites, mapping):
    mapping.update(getMappingFromOntology(ontology = "Disease", source = config.HMDB_DO_source))
    relationships = defaultdict(list)
    associations = iconfig.HMDB_associations
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

def build_HMDB_dictionary(metabolites):
    directory = os.path.join(dbconfig.databasesDir,"HMDB")
    filename = iconfig.HMDB_dictionary_file
    outputfile = os.path.join(directory, filename)
    
    with open(outputfile, 'w') as out:
        for metid in metabolites:
            if "name" in metabolites[metid]:
                name = metabolites[metid]["name"]
                out.write(metid+"\t"+name.lower()+"\n")
            if "synonyms" in metabolites[metid]:
                for synonym in metabolites[metid]["synonyms"]:
                    out.write(metid+"\t"+synonym.lower()+"\n")

