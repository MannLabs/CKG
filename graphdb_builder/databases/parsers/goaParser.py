import pandas as pd
import gzip
from collections import defaultdict
from graphdb_builder import builder_utils
import mapping as mp

def parser(database_dir, download=True):
    cwd = os.path.abspath(os.path.dirname(__file__))
    config = ckg_utils.get_configuration(os.path.join(cwd, '../config/goaConfig.yml'))
    url = config['url']
    rel_header = config['header']
    
    protein_mapping = mp.getMappingForEntity(entity="Protein")    
    valid_proteins = list(set(protein_mapping.values))

    directory = os.path.join(databases_directory,"GOA")
    builder_utils.checkDirectory(directory)
    file_name = os.path.join(directory, url.split('/')[-1])
    if download:
        builder_utils.downloadDB(url, directory)

    annotations = parse_annotations_with_panda(file_name, valid_proteins)

    return annotations, header

def parse_annotations_with_pandas(annotation_file, valid_proteins=None):
    roots = {'F':'Molecular_function', 'C':'Cellular_component', 'P':'Biological_process'}
    selected_columns = [0, 1, 4, 6, 8, 14]
    new_columns = ['source', 'START_ID', 'END_ID', 'evidence', 'root', 'original_source']
    annotations = defaultdict(set)
    annotations_df = pd.DataFrame(columns=new_columns)
    index = 'START_ID'
    chunksize = 10 ** 6
    for chunk in pd.read_csv(annotation_file, chunksize=chunksize, sep='\t', comment='!', header=None, low_memory=False):
        data = chunk[selected_columns]
        data.columns = new_columns
        data = data.set_index(index)
        good_keys = data.index.intersection(valid_proteins)
        if len(good_keys) >= 1:
            data = data.loc[good_keys]
            data['root'] = [roots[r] for r in data['root'].values]
            annotations_df = annotations_df.append(data.reset_index(), ignore_index=True)
    for name, group in annotations_df.groupby('root'):
        group['TYPE'] = "ASSOCIATED_WITH"
        group = group[['START_ID', 'END_ID', 'TYPE', 'evidence','source']]
        annotations[name] = group.to_records(index=False).tolist()
    return annotations
            
if __name__ == '__main__':
    annotation_file = '../../data/databases/GOA/goa_uniprot_all.gaf.gz'

    annotations, header = parse_annotations_with_pandas(annotation_file, valid_proteins=proteins)

