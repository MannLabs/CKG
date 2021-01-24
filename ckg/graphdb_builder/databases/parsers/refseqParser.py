import os.path
from collections import defaultdict
from ckg.graphdb_builder import builder_utils

#########################
#          RefSeq       #
#########################
def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="refseqConfig.yml", data_type='databases')
    url = config['refseq_url']
    ftp_dir = config['refseq_ftp_dir']
    entities = defaultdict(set)
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory, "RefSeq")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    headers = config['headerEntities']
    taxid = 9606

    if download:
        file_dir = builder_utils.list_ftp_directory(ftp_dir)[0]
        new_file = file_dir.split('/')[-1]+"_feature_table.txt.gz"
        url = ftp_dir + file_dir.split('/')[-1] + "/" + new_file
        builder_utils.downloadDB(url, directory)
        fileName = os.path.join(directory, new_file)

    if os.path.isfile(fileName):
        df = builder_utils.read_gzipped_file(fileName)
        first = True
        for line in df:
            if first:
                first = False
                continue
            data = line.rstrip("\r\n").split("\t")
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

    builder_utils.remove_directory(directory)

    return (entities, relationships, headers)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base, "../../../../data/databases")
    parser(db_path)
