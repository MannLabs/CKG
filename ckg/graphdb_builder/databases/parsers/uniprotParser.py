import os.path
from collections import defaultdict
import pandas as pd
import re
from ckg.graphdb_builder import mapping as mp, builder_utils

#########################
#       UniProt         #
#########################
def parser(databases_directory, import_directory, download=True, updated_on=None):
    config = builder_utils.get_config(config_name="uniprotConfig.yml", data_type='databases')
    relationships_header = config['relationships_header']
    parse_release_notes(databases_directory, config, download)
    stats = parse_fasta(databases_directory, config, import_directory, download=download, updated_on=updated_on)
    #Proteins
    stats.update(parse_idmapping_file(databases_directory, config, import_directory, download=download, updated_on=updated_on))
    #Peptides
    entities, relationships = parseUniProtPeptides(config, databases_directory, download)
    entities_header = config['peptides_header']
    output_file = os.path.join(import_directory, "Peptide.tsv")
    stats.update(print_single_file(entities, entities_header, output_file, "entity", "Peptide", is_first=True, updated_on=updated_on))
    stats.update(print_multiple_relationships_files(relationships, relationships_header, import_directory, is_first=True, updated_on=updated_on))

    #Variants
    stats.update(parseUniProtVariants(config, databases_directory, import_directory, download))

    #Gene ontology annotation
    relationships = parseUniProtAnnotations(config, databases_directory, download)
    relationships_header = config['go_header']
    stats.update(print_multiple_relationships_files(relationships, relationships_header, import_directory, is_first=True, updated_on=updated_on))

    directory = os.path.join(databases_directory, "UniProt")
    builder_utils.remove_directory(directory)

    return stats


def parse_release_notes(databases_directory, config, download=True):
    release_notes_url = config['release_notes']
    directory = os.path.join(databases_directory, "UniProt")
    builder_utils.checkDirectory(directory)
    if download:
        builder_utils.downloadDB(release_notes_url, directory)


def parse_fasta(databases_directory, config, import_directory, download=True, updated_on=None):
    stats = set()
    url = config['uniprot_fasta_file']
    entities_output_file = os.path.join(import_directory, "Amino_acid_sequence.tsv")
    rel_output_file = os.path.join(import_directory, "Protein_HAS_Sequence_Amino_acid_sequence.tsv")

    directory = os.path.join(databases_directory, "UniProt")
    builder_utils.checkDirectory(directory)
    file_name = os.path.join(directory, url.split('/')[-1])

    if download:
        builder_utils.downloadDB(url, directory)

    ff = builder_utils.read_gzipped_file(file_name)
    records = builder_utils.parse_fasta(ff)
    num_entities = 0
    with open(entities_output_file, 'w', encoding='utf-8') as ef:
        ef.write('ID\theader\tsequence\tsize\tsource\n')
        with open(rel_output_file, 'w', encoding='utf-8') as rf:
            rf.write('START_ID\tEND_ID\tTYPE\tsource\n')
            for i, batch in enumerate(builder_utils.batch_iterator(records, 1000)):
                for record in batch:
                    identifier = record.id.split('|')[1]
                    header = record.id
                    sequence = str(record.seq)
                    sequence_len = len(str(sequence))
                    ef.write(identifier+"\t"+header+'\t'+sequence+'\t'+str(sequence_len)+'\tUniProt\n')
                    rf.write(identifier+'\t'+identifier+'\tHAS_SEQUENCE\tUniProt\n')
                    num_entities += 1

    stats.add(builder_utils.buildStats(num_entities, "entity", "Amino_acid_sequence", "UniProt", entities_output_file, updated_on))
    stats.add(builder_utils.buildStats(num_entities, "relationships", "HAS_SEQUENCE", "UniProt", rel_output_file, updated_on))

    return stats


def parse_idmapping_file(databases_directory, config, import_directory, download=True, updated_on=None):
    regex_transcript = r"(-\d+$)"
    taxids = config['species']

    proteins_output_file = os.path.join(import_directory, "Protein.tsv")
    pdbs_output_file = os.path.join(import_directory, "Protein_structures.tsv")
    proteins = {}

    url = config['uniprot_id_url']
    directory = os.path.join(databases_directory, "UniProt")
    builder_utils.checkDirectory(directory)
    file_name = os.path.join(directory, url.split('/')[-1])
    mapping_file = os.path.join(directory, 'mapping.tsv')
    if download:
        builder_utils.downloadDB(url, directory)

    fields = config['uniprot_ids']
    synonymFields = config['uniprot_synonyms']

    identifier = None
    transcripts = set()
    skip = set()
    is_first = True
    uf = builder_utils.read_gzipped_file(file_name)
    aux = {}
    stats = set()
    mp.reset_mapping(entity="Protein")
    with open(mapping_file, 'w', encoding='utf-8') as out:
        for line in uf:
            data = line.rstrip("\r\n").split("\t")
            iid = data[0]
            if 'UniParc' in data:
                if re.search(regex_transcript, iid):
                    transcripts.add(iid)
                continue
            # data = line.decode('utf-8').rstrip("\r\n").split("\t")
            # iid = data[0]
            field = data[1]
            alias = data[2]

            if iid not in skip:
                skip = set()
                if re.search(regex_transcript, iid):
                    transcripts.add(iid)
                if iid not in aux and iid.split('-')[0] not in aux:
                    if identifier is not None:
                        prot_info["synonyms"] = synonyms
                        aux[identifier].update(prot_info)
                        if "UniProtKB-ID" in aux[identifier] and "NCBI_TaxID" in aux[identifier]:
                            for synonym in synonyms:
                                out.write(identifier+"\t"+synonym+"\n")
                            proteins[identifier] = aux[identifier]
                            for t in transcripts:
                                proteins[t] = aux[identifier]
                                for synonym in synonyms:
                                    out.write(t+"\t"+synonym+"\n")
                            if len(transcripts) > 0:
                                proteins[identifier].update({"isoforms": transcripts})
                                transcripts = set()
                            aux.pop(identifier, None)
                            if len(proteins) >= 1000:
                                entities, relationships, pdb_entities = format_output(proteins)
                                stats.update(print_single_file(entities, config['proteins_header'], proteins_output_file, "entity", "Protein", is_first, updated_on))
                                stats.update(print_single_file(pdb_entities, config['pdb_header'], pdbs_output_file, "entity", "Protein_structure", is_first, updated_on))
                                stats.update(print_multiple_relationships_files(relationships, config['relationships_header'], import_directory, is_first, updated_on))
                                is_first = False
                                proteins = {}
                    identifier = iid
                    transcripts = set()
                    aux[identifier] = {}
                    prot_info = {}
                    synonyms = []
                if field in fields:
                    if field == 'NCBI_TaxID':
                        if int(alias) not in taxids:
                            skip.add(identifier)
                            aux.pop(identifier, None)
                            identifier = None
                    if field in synonymFields:
                        synonyms.append(alias)
                    prot_info.setdefault(field, [])
                    prot_info[field].append(alias)

        uf.close()

    if len(proteins) > 0:
        entities, relationships, pdb_entities = format_output(proteins)
        stats.update(print_single_file(entities, config['proteins_header'], proteins_output_file, "entity", "Protein", is_first, updated_on))
        stats.update(print_single_file(pdb_entities, config['pdb_header'], pdbs_output_file, "entity", "Protein_structure", is_first, updated_on))
        stats.update(print_multiple_relationships_files(relationships, config['relationships_header'], import_directory, is_first, updated_on))

    mp.mark_complete_mapping(entity="Protein")

    return stats


def format_output(proteins):
    entities = set()
    relationships = defaultdict(set)
    pdb_entities = set()
    for protein in proteins:
        accession = ""
        name = ""
        synonyms = []
        description = ""
        taxid = None
        pdb = []
        if "UniProtKB-ID" in proteins[protein]:
            accession  = proteins[protein]["UniProtKB-ID"][0]
        if "Gene_Name" in proteins[protein]:
            name = proteins[protein]["Gene_Name"][0]
            for i in proteins[protein]["Gene_Name"]:
                relationships[("Protein","GENE_TRANSLATED_INTO")].add((i, protein, "GENE_TRANSLATED_INTO", 'UniProt'))
        if "RefSeq" in proteins[protein]:
            for i in proteins[protein]["RefSeq"]:
                relationships[("Protein","TRANSCRIPT_TRANSLATED_INTO")].add((i, protein, "TRANSCRIPT_TRANSLATED_INTO", 'UniProt'))
        if "synonyms" in proteins[protein]:
            synonyms = proteins[protein]["synonyms"]
        if "NCBI_TaxID" in proteins[protein]:
            taxid = proteins[protein]["NCBI_TaxID"][0]
            relationships[("Protein", "BELONGS_TO_TAXONOMY")].add((protein, int(taxid), "BELONGS_TO_TAXONOMY", 'UniProt'))
        if 'PDB' in proteins[protein]:
            pdb = proteins[protein]['PDB']
            for i in pdb:
                pdb_entities.add((i, 'Protein_structure', 'Uniprot', 'http://www.rcsb.org/structure/{}'.format(i)))
                relationships[("Protein","HAS_STRUCTURE")].add((protein, i, "HAS_STRUCTURE", 'UniProt'))
        if "description" in proteins[protein]:
            description = proteins[protein]["description"]
        if "isoforms" in proteins[protein]:
            for i in proteins[protein]['isoforms']:
                relationships[('Transcript','IS_ISOFORM')].add((i, protein, 'IS_ISOFORM', 'UniProt'))

        entities.add((protein, "Protein", accession, name, ",".join(synonyms), description, int(taxid)))


    return entities, relationships, pdb_entities

def print_single_file(data, header, output_file, data_type, data_object, is_first, updated_on):
    stats = set()
    df = pd.DataFrame(list(data), columns=header)
    stats.add(builder_utils.buildStats(len(data), data_type, data_object, "UniProt", output_file, updated_on))
    with open(output_file, 'a', encoding='utf-8') as ef:
        df.to_csv(path_or_buf=ef, sep='\t',
                header=is_first, index=False, quotechar='"',
                line_terminator='\n', escapechar='\\')

    return stats

def print_multiple_relationships_files(data, header, output_dir, is_first, updated_on):
    stats = set()
    for entity, relationship in data:
        df = pd.DataFrame(list(data[(entity, relationship)]), columns=header)
        output_file = os.path.join(output_dir, entity+"_"+relationship.lower() + ".tsv")
        stats.add(builder_utils.buildStats(len(data[(entity, relationship)]), 'relationships', relationship, "UniProt", output_file, updated_on))
        with open(output_file, 'a', encoding='utf-8') as ef:
            df.to_csv(path_or_buf=ef, sep='\t',
            header=is_first, index=False, quotechar='"',
            line_terminator='\n', escapechar='\\')

    return stats

def addUniProtTexts(textsFile, proteins):
    with open(textsFile, 'r', encoding='utf-8') as tf:
        for line in tf:
            data = line.rstrip("\r\n").split("\t")
            protein = data[0]
            name = data[1]
            function = data[3]

            if protein in proteins:
                proteins[protein].update({"description": function})


def parseUniProtVariants(config, databases_directory, import_directory, download=True, updated_on=None):
    data = defaultdict()
    variant_regex = r"(g\.\w+>\w)"
    chromosome_regex = r"(\w+)[p|q]"
    url = config['uniprot_variant_file']
    aa = config['amino_acids']
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory, "UniProt")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)

    vf = builder_utils.read_gzipped_file(fileName)
    din = False
    i = 0
    stats = set()
    is_first = True
    for line in vf:
        line = line
        if not line.startswith('#') and not din:
            continue
        elif i <= 2:
            din = True
            i += 1
            continue
        data = line.rstrip("\r\n").split("\t")
        if len(data) > 9:
            gene = data[0]
            protein = data[1]
            pvariant = data[2]
            externalID = data[3]
            impact = data[4]
            clin_relevance = data[5]
            disease = data[6]
            chromosome_coord = data[8]
            original_source = data[13]
            ref = pvariant[2:5]
            pos = pvariant[5:-3]
            alt = pvariant[-3:]
            var_matches = re.search(variant_regex, data[9])
            chr_matches = re.search(chromosome_regex, chromosome_coord)
            if var_matches and chr_matches:
                chromosome = 'chr'+chr_matches.group(1)
                ident = chromosome+":"+var_matches.group(1)
                altName = [externalID, data[5], pvariant, chromosome_coord]
                if ref in aa and alt in aa:
                    altName.append(aa[ref]+pos+aa[alt])
                pvariant = protein+"_"+pvariant
                entities.add((ident, "Known_variant", pvariant, externalID, ",".join(altName), impact, clin_relevance, disease, original_source, "UniProt"))
                if chromosome != 'chr-':
                    relationships[('Chromosome', 'known_variant_found_in_chromosome')].add((ident, chromosome.replace('chr',''), "VARIANT_FOUND_IN_CHROMOSOME","UniProt"))
                if gene != "":
                    relationships[('Gene','known_variant_found_in_gene')].add((ident, gene, "VARIANT_FOUND_IN_GENE", "UniProt"))
                if protein !="":
                    relationships[('Protein','known_variant_found_in_protein')].add((ident, protein, "VARIANT_FOUND_IN_PROTEIN", "UniProt"))

                if len(entities) >= 1000:
                    stats.update(print_single_file(entities, config['variants_header'], os.path.join(import_directory, "Known_variant.tsv"), "entity", "Known_variant", is_first, updated_on))
                    stats.update(print_multiple_relationships_files(relationships, config['relationships_header'], import_directory, is_first, updated_on))
                    entities = set()
                    relationships = defaultdict(set)

                    is_first = False

    if len(entities) > 0:
        stats.update(print_single_file(entities, config['variants_header'], os.path.join(import_directory, "Known_variant.tsv"), "entity", "Known_variant", is_first, updated_on))
        stats.update(print_multiple_relationships_files(relationships, config['relationships_header'], import_directory, is_first, updated_on))
        del(entities)
        del(relationships)

    return stats

def parseUniProtAnnotations(config, databases_directory, download=True):
    roots = {'F':'Molecular_function', 'C':'Cellular_component', 'P':'Biological_process'}
    url = config['uniprot_go_annotations']
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"UniProt")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)

    af = builder_utils.read_gzipped_file(fileName)
    for line in af:
        line = line
        if line.startswith('!'):
            continue
        data = line.rstrip("\r\n").split("\t")
        identifier = data[1]
        go = data[4]
        evidence = data[6]
        root = data[8]
        if root in roots:
            root = roots[root]
            relationships[(root,'associated_with')].add((identifier, go, "ASSOCIATED_WITH", evidence, 5, "UniProt"))

    return relationships

def parseUniProtPeptides(config, databases_directory, download=True):
    file_urls = config['uniprot_peptides_files']
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"UniProt")
    builder_utils.checkDirectory(directory)
    for url in file_urls:
        fileName = os.path.join(directory, url.split('/')[-1])
        if download:
            builder_utils.downloadDB(url, directory)
        first = True
        with open(fileName, 'r', encoding='utf-8') as f:
            for line in f:
                if first:
                    first = False
                    continue

                data = line.rstrip("\r\n").split("\t")
                peptide = data[0]
                accs = data[6].split(",")
                is_unique = True
                if len(accs) > 1:
                    is_unique = False
                entities.add((peptide, "Peptide", "tryptic peptide", is_unique))
                for protein in accs:
                    relationships[("Peptide", 'belongs_to_protein')].add((peptide, protein, "BELONGS_TO_PROTEIN", "UniProt"))
    return entities, relationships

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base, "../../../../data/databases")
    import_path = os.path.join(base, "../../../../data/imports/databases")
    parser(databases_directory=db_path, import_directory=import_path, download=False)
