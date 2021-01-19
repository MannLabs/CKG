import os.path
import re
from collections import defaultdict
from ckg.graphdb_builder import builder_utils


def parser(databases_directory, download=True):
    config = builder_utils.get_config(config_name="gwasCatalogConfig.yml", data_type='databases')
    url = config['GWASCat_url']
    entities_header = config['entities_header']
    relationships_header = config['relationships_header']
    entities = set()
    relationships = defaultdict(set)
    directory = os.path.join(databases_directory, "GWAScatalog")
    builder_utils.checkDirectory(directory)
    fileName = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)
    with open(fileName, 'r', encoding="utf-8") as catalog:
        for line in catalog:
            data = line.rstrip("\r\n").split("\t")
            if len(data) > 36:
                pubmedid = data[1]
                date = data[3]
                title = data[6]
                sample_size = data[8]
                replication_size = data[9]
                #chromosome = data[11]
                #position = data[12]
                #genes_mapped = data[14].split(" - ")
                snp_id = data[20].split('-')[0]
                freq = data[26]
                pval = data[27]
                odds_ratio = data[30]
                trait = data[34]
                exp_factor = data[35]
                study = data[36]

                entities.add((study, "GWAS_study", title, date, sample_size, replication_size, trait))
                if pubmedid != "":
                    relationships["published_in_publication"].add((study, pubmedid, "PUBLISHED_IN", "GWAS Catalog"))
                if snp_id != "":
                    relationships["variant_found_in_gwas"].add((re.sub(r"^\W+|\W+$", "",snp_id), study, "VARIANT_FOUND_IN_GWAS", freq, pval, odds_ratio, trait, "GWAS Catalog"))
                if exp_factor != "":
                    exp_factor = exp_factor.split('/')[-1].replace('_', ':')
                    relationships["studies_trait"].add((study, exp_factor, "STUDIES_TRAIT", "GWAS Catalog"))

    builder_utils.remove_directory(directory)

    return (entities, relationships, entities_header, relationships_header)
