import os.path
from collections import defaultdict
from lxml import etree
import zipfile
from ckg.graphdb_builder import mapping as mp, builder_utils

#########################
#       Drug Bank       #
#########################
def parser(databases_directory):
    config = builder_utils.get_config(config_name="drugBankConfig.yml", data_type='databases')
    directory = os.path.join(databases_directory, "DrugBank")
    builder_utils.checkDirectory(directory)
    drugs = extract_drugs(config, directory)
    build_DrugBank_dictionary(config, directory, drugs)
    relationships = build_relationships_from_DrugBank(config, drugs)
    entities, attributes = build_drug_entity(config, drugs)
    entities_header = ['ID'] + attributes
    relationships_headers = config['relationships_headers']

    return (entities, relationships, entities_header, relationships_headers)


def extract_drugs(config, directory):
    drugs = {}
    prefix = '{http://www.drugbank.ca}'
    url = config['DrugBank_url']
    fileName = os.path.join(directory, url.split('/')[-1])
    fields = config['DrugBank_fields']
    attributes = config['DrugBank_attributes']
    parentFields = config['DrugBank_parentFields']
    structuredFields = config['DrugBank_structures']

    vocabulary = parseDrugBankVocabulary(config, directory)
    with zipfile.ZipFile(fileName, 'r') as zipped:
        for zfile in zipped.namelist():
            zipped.extract(member=zfile, path=directory)
            xfile = os.path.join(directory, zfile)
            with open(xfile, 'rb') as f:
                context = etree.iterparse(f, events=("end",), tag=prefix+"drug")
                for a, elem in context:
                    synonyms = set()
                    values = {child.tag.replace(prefix, ''): child.text for child in elem.iterchildren() if child.tag.replace(prefix, '') in fields and child.text is not None}
                    if "drugbank-id" in values:
                        synonyms.add(values["drugbank-id"])
                    for child in elem.iterchildren():
                        if child.tag.replace(prefix, '') in parentFields:
                            label = child.tag.replace(prefix, '')
                            values[label] = []
                            for intchild in child.iter():
                                if intchild.text is not None and intchild.text.strip() != "":
                                    if label in structuredFields:
                                        if intchild.tag.replace(prefix, '') in structuredFields[label]:
                                            if label == "external-identifiers":
                                                synonyms.add(intchild.text)
                                            else:
                                                values[label].append(intchild.text)
                                    elif intchild.tag.replace(prefix, '') in fields and intchild.text:
                                        values[label].append(intchild.text)
                                    elif intchild.tag.replace(prefix, '') in attributes and intchild.text:
                                        values[intchild.tag.replace(prefix, '')] = intchild.text

                    if "drugbank-id" in values and len(values) > 2:
                        if values["drugbank-id"] in vocabulary:
                            values["id"] = vocabulary[values["drugbank-id"]]
                            synonyms.add(values["drugbank-id"])
                            #values["alt_drugbank-id"] = vocabulary[values['id']]
                            values["synonyms"] = list(synonyms)
                            drugs[values["id"]] = values
    return drugs


def parseDrugBankVocabulary(config, directory):
    vocabulary = {}
    url = config['DrugBank_vocabulary_url']
    fileName = os.path.join(directory, url.split('/')[-1])
    with zipfile.ZipFile(fileName, 'r') as zipped:
        for f in zipped.namelist():
            with zipped.open(f) as vf:
            # with open(os.path.join(directory,f), 'r') as vf:
                for line in vf:
                    data = line.decode('utf-8').rstrip('\r\n').split(',')
                    primary = data[0]
                    secondaries = data[1].split(' | ')
                    for sec in secondaries:
                        vocabulary[sec] = primary
                        vocabulary[primary] = primary
    return vocabulary


def build_relationships_from_DrugBank(config, drugs):
    relationships = defaultdict(list)
    associations = config['DrugBank_associations']
    for did in drugs:
        for ass in associations:
            ident = ass
            if len(associations[ass]) > 1:
                ident = associations[ass][1]
            if ass in drugs[did]:
                if type(drugs[did][ass]) == list:
                    partners = drugs[did][ass]
                    if ass == "drug-interactions":
                        partners = zip(partners[0::2], partners[1::2])
                    elif ass in ["snp-effects", 'snp-adverse-drug-reactions']:
                        partners = zip(partners[0::3], partners[1::3], partners[2::3])
                    elif ass == "targets":
                        partners = zip(partners[0::2], partners[1::2])
                        partners = [p for r, p in partners if r == "UniProtKB"]
                    for partner in partners:
                        rel = (did, partner, associations[ass][0], "DrugBank")
                        relationships[ident].append(tuple(builder_utils.flatten(rel)))
                else:
                    partner = drugs[did][ass]
                    relationships[ident].append((did, partner, associations[ass][0], "DrugBank"))

    return relationships


def build_drug_entity(config, drugs):
    entities = set()
    attributes = config['DrugBank_attributes']
    properties = config['DrugBank_exp_prop']
    allAttr = attributes[:-1] + [p.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for p in properties]
    for did in drugs:
        entity = []
        entity.append(did)
        for attr in attributes:
            if attr in drugs[did]:
                if type(drugs[did][attr]) == list:
                    if attr == "experimental-properties":
                        newAttr  = dict(zip(drugs[did][attr][0::2], drugs[did][attr][1::2]))
                        for prop in properties:
                            if prop in newAttr:
                                entity.append(newAttr[prop])
                            else:
                                entity.append('')
                    else:
                        lattr = "|".join(drugs[did][attr])
                        entity.append(lattr)
                else:
                    entity.append(drugs[did][attr])
            else:
                entity.append('')
        entities.add(tuple(entity))

    return entities, allAttr


def build_DrugBank_dictionary(config, directory, drugs):
    filename = config['DrugBank_dictionary_file']
    outputfile = os.path.join(directory, filename)
    mp.reset_mapping(entity="Drug")
    with open(outputfile, 'w', encoding='utf-8') as out:
        for did in drugs:
            if "name" in drugs[did]:
                name = drugs[did]["name"]
                out.write(did+"\t"+name.lower()+"\n")
            if "synonyms" in drugs[did]:
                for synonym in drugs[did]["synonyms"]:
                    out.write(did+"\t"+synonym.lower()+"\n")

    mp.mark_complete_mapping(entity="Drug")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base, "../../../../data/databases")
    parser(db_path)
