import os.path
from KnowledgeGrapher.databases import databases_config as dbconfig
from KnowledgeGrapher.databases.config import drugBankConfig as iconfig
from collections import defaultdict
from lxml import etree
import zipfile
from KnowledgeGrapher import utils


#########################
#       Drug Bank       #
#########################
def parser():
    drugs = extract_drugs()
    build_DrugBank_dictionary(drugs)
    relationships = build_relationships_from_DrugBank(drugs)
    entities, attributes = build_drug_entity(drugs)
    entities_header = ['ID'] + attributes
    relationships_headers = iconfig.relationships_headers
    
    return (entities, relationships, entities_header, relationships_headers)

def extract_drugs():
    drugs = {}
    prefix = '{http://www.drugbank.ca}'
    relationships = set()
    url = iconfig.DrugBank_url
    directory = os.path.join(dbconfig.databasesDir,"DrugBank")
    utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    fields = iconfig.DrugBank_fields
    parentFields = iconfig.DrugBank_parentFields
    structuredFields = iconfig.DrugBank_structures
    vocabulary = parseDrugBankVocabulary()
    i = 1
    with zipfile.ZipFile(fileName, 'r') as zipped:
        for f in zipped.namelist():
            data = zipped.read(f) 
            root = etree.fromstring(data)
            context = etree.iterwalk(root, events=("end",), tag=prefix+"drug")
            for a,elem in context:
                synonyms = set()
                values = {child.tag.replace(prefix,''):child.text for child in elem.iterchildren() if child.tag.replace(prefix,'') in fields and child.text is not None}
                if "drugbank-id" in values:
                    synonyms.add(values["drugbank-id"])
                for child in elem.iterchildren(): 
                    if child.tag.replace(prefix,'') in parentFields:
                        label = child.tag.replace(prefix,'')
                        values[label] = []
                        for intchild in child.iter():
                            if intchild.text is not None and intchild.text.strip() != "":
                                if label in structuredFields:
                                    if intchild.tag.replace(prefix,'') in structuredFields[label]:
                                        if label == "external-identifiers":
                                            synonyms.add(intchild.text)
                                        else:
                                            values[label].append(intchild.text) 
                                elif intchild.tag.replace(prefix,'') in fields and intchild.text:
                                    values[label].append(intchild.text) 
                
                if "drugbank-id" in values and len(values) > 2:
                    if values["drugbank-id"] in vocabulary:
                        values["id"] = vocabulary[values["drugbank-id"]]
                        synonyms.add(values["drugbank-id"])
                        #values["alt_drugbank-id"] = vocabulary[values['id']]
                        values["synonyms"] = list(synonyms)
                        drugs[values["id"]] = values

    return drugs

def parseDrugBankVocabulary():
    vocabulary = {}
    url = iconfig.DrugBank_vocabulary_url
    directory = os.path.join(dbconfig.databasesDir,"DrugBank")
    fileName = os.path.join(directory, url.split('/')[-1])
    with zipfile.ZipFile(fileName, 'r') as zipped:
        for f in zipped.namelist():
            with open(os.path.join(directory,f), 'r') as vf:
                for line in vf:
                    data = line.rstrip('\r\n').split(',')
                    primary = data[0]
                    secondaries = data[1].split(' | ')
                    for sec in secondaries:
                        vocabulary[sec] = primary
                        vocabulary[primary] = primary 
    return vocabulary


def build_relationships_from_DrugBank(drugs):
    relationships = defaultdict(list)
    associations = iconfig.DrugBank_associations
    for did in drugs:
        for ass in associations:
            ident = ass
            if len(associations[ass]) > 1:
                ident = associations[ass][1]
            if ass in drugs[did]:
                if type(drugs[did][ass]) == list:
                    partners = drugs[did][ass]
                    if ass == "drug-interactions":
                        partners = zip(partners[0::2],partners[1::2])
                    elif ass in ["snp-effects", 'snp-adverse-drug-reactions']:
                        partners = zip(partners[0::3],partners[1::3],partners[2::3])
                    elif ass == "targets":
                        partners = zip(partners[0::2],partners[1::2])
                        partners = [p for r,p in partners if r == "UniProtKB"]
                    for partner in partners:
                        rel = (did, partner, associations[ass][0], "DrugBank")
                        relationships[ident].append(tuple(utils.flatten(rel)))
                else:
                    partner = drugs[did][ass]
                    relationships[ident].append((did, partner, associations[ass][0], "DrugBank"))
    
    return relationships

def build_drug_entity(drugs):
    entities = set()
    attributes = iconfig.DrugBank_attributes
    properties = iconfig.DrugBank_exp_prop
    allAttr = attributes + [p.replace(' ','_') for p in properties]
    for did in drugs:
        entity = []
        entity.append(did)
        for attr in attributes:
            if attr in drugs[did]:
                if type(drugs[did][attr]) == list:
                    if attr == "experimental-properties":
                        newAttr  = dict(zip(drugs[did][attr][0::2],drugs[did][attr][1::2]))
                        for prop in properties:
                            if prop in newAttr:
                                entity.append(newAttr[prop])
                            else:
                                entity.append('')
                    else:
                        lattr = ";".join(drugs[did][attr])
                        entity.append(lattr)
                else:
                    entity.append(drugs[did][attr])
            else:
                entity.append('')
        entities.add(tuple(entity))
    
    return entities, allAttr

def build_DrugBank_dictionary(drugs):
    directory = os.path.join(dbconfig.databasesDir,"DrugBank")
    filename = iconfig.DrugBank_dictionary_file
    outputfile = os.path.join(directory, filename)
    
    with open(outputfile, 'w') as out:
        for did in drugs:
            if "name" in drugs[did]:
                name = drugs[did]["name"]
                out.write(did+"\t"+name.lower()+"\n")
            if "synonyms" in drugs[did]:
                for synonym in drugs[did]["synonyms"]:
                    out.write(did+"\t"+synonym.lower()+"\n")
