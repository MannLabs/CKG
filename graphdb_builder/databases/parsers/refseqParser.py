import os.path
import gzip
from collections import defaultdict
import ckg_utils
from graphdb_builder import builder_utils

#########################
#          RefSeq       # 
#########################
def parser(databases_directory, download=True):
    config = ckg_utils.get_configuration('../databases/config/refseqConfig.yml')
    url = config['refseq_url']
    entities = defaultdict(set)
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory,"RefSeq")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    headers = config['headerEntities']
    taxid = 9606
    
    if download:
        builder_utils.downloadDB(url, directory)

    df = gzip.open(fileName, 'r')
    first = True
    for line in df:
        if first:
            first = False
            continue
        data = line.decode('utf-8').rstrip("\r\n").split("\t")
        tclass = data[1]
        assembly = data[2]
        chrom = data[5]
        geneAcc = data[6]
        start = data[7]
        end = data[8]
        strand = data[9]
        protAcc = data[10]
        name = data[13]
        symbol = data[14]
        
        if protAcc != "":
            entities["Transcript"].add((protAcc, "Transcript", name, tclass, assembly, taxid))
            if chrom != "":
                entities["Chromosome"].add((chrom, "Chromosome", chrom, taxid))
                relationships["LOCATED_IN"].add((protAcc, chrom, "LOCATED_IN", start, end, strand, "RefSeq"))
            if symbol != "":
                relationships["TRANSCRIBED_INTO"].add((symbol, protAcc, "TRANSCRIBED_INTO", "RefSeq"))
        elif geneAcc != "":
            entities["Transcript"].add((geneAcc, "Transcript", name, tclass, assembly, taxid))
            if chrom != "":
                entities["Chromosome"].add((chrom, "Chromosome", chrom, taxid))
                relationships["LOCATED_IN"].add((protAcc, chrom, "LOCATED_IN", start, end, strand, "RefSeq"))
    df.close()

    return (entities, relationships, headers)
