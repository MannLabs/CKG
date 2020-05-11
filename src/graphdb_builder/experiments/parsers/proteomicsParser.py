import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from graphdb_builder import builder_utils, mapping


def parser(projectId, directory=None):
    data_type = 'proteomics'
    data = {}
    cwd = os.path.abspath(os.path.dirname(__file__))
    config = builder_utils.get_config(config_name="proteomics.yml", data_type='experiments')

    if directory is None:
        directory = os.path.join(cwd, '../../../../data/experiments/PROJECTID/' + data_type)
        if 'directory' in config:
            directory = os.path.join(cwd, config['directory'] + data_type)
    directory = directory.replace('PROJECTID', projectId)
    data = parse_from_directory(projectId, directory, config)

    return data


def parse_from_directory(projectId, directory, configuration=None):
    data = {}
    processing_results = [x[0] for x in os.walk(directory)]
    for results_path in processing_results:
        processing_tool = results_path.split('/')[-1]
        if processing_tool in configuration:
            sample_mapping = mapping.get_mapping_analytical_samples(projectId)
            if len(sample_mapping) > 0:
                mapping.map_experiment_files(projectId, os.path.join(directory, processing_tool), sample_mapping)
            tool_configuration = configuration[processing_tool]
            for dtype in tool_configuration:
                dataset_configuration = tool_configuration[dtype]
                missing_conf = check_minimum_configuration(dataset_configuration)
                if len(missing_conf) == 0:
                    dfile_regex = re.compile(dataset_configuration['file'])
                    filepath = ''
                    for dir_content in os.walk(results_path):
                        for f in dir_content[2]:
                            if dfile_regex.match(f):
                                filepath = os.path.join(results_path, f)
                                break
                        data.update(parser_from_file(file_path=filepath, configuration=dataset_configuration, data_type=dtype))
                else:
                    raise Exception("Error when importing proteomics experiment.\n Missing configuration: {}".format(",".join(missing_conf)))

    return data


def parser_from_file(file_path, configuration, data_type, is_standard=True):
    data = {}
    if is_standard:
        df = parse_standard_dataset(file_path, configuration)
    else:
        df = parse_dataset(file_path, configuration)

    if df is not None and not df.empty:
        if data_type == "proteins":
            data[(data_type, 'w')] = extract_protein_subject_rels(df, configuration)
        elif data_type == "peptides":
            data[('subject_peptide', 'w')] = extract_peptide_subject_rels(df, configuration)
            data[('peptide_protein', 'w')] = extract_peptide_protein_rels(df, configuration)
            data[(data_type, 'w')] = extract_peptides(df, configuration)
        else:
            data[('modifiedprotein_subject', 'a')] = extract_protein_modification_subject_rels(df, configuration)
            data[('modifiedprotein_protein', 'a')] = extract_protein_protein_modification_rels(df, configuration)
            data[('modifiedprotein_peptide', 'a')] = extract_peptide_protein_modification_rels(df, configuration)
            data[('modifiedprotein', 'a')] = extract_protein_modifications_rels(df, configuration)
            data[('modifiedprotein_modification', 'a')] = extract_protein_modifications_modification_rels(df, configuration)

    return data


def update_configuration(data_type, processing_tool, value_col='LFQ intensity', columns=[]):
    configuration = {}
    if processing_tool is not None:
        config = builder_utils.get_config(config_name="proteomics.yml", data_type='experiments')
        if processing_tool in config:
            tool_configuration = config[processing_tool]
            if data_type in tool_configuration:
                configuration = tool_configuration[data_type]
                configuration['columns'].extend(columns)
                configuration['valueCol'] = value_col

    return configuration


def parse_dataset(filepath, configuration):
    data = None
    if os.path.isfile(filepath):
        data, regex = load_dataset(filepath, configuration)
        if data is not None:
            if 'log' in configuration:
                log = configuration['log']
                cols = get_value_cols(data, configuration)
                if log == 'log2':
                    data[cols] = np.log2(data[cols]).replace([np.inf, -np.inf], np.nan)
                elif log == 'log10':
                    data[cols] = np.log10(data[cols]).replace([np.inf, -np.inf], np.nan)
    return data


def parse_standard_dataset(file_path, configuration):
    dataset = None
    if os.path.isfile(file_path):
        data, regex = load_dataset(file_path, configuration)
        if data is not None:
            log = configuration['log']
            combine = 'regex'
            if 'combine' in configuration:
                combine = configuration['combine']
            if combine == 'valueCol':
                value_cols = get_value_cols(data, configuration)
                subjectDict = extract_subject_replicates(data, value_cols)
            else:
                subjectDict = extract_subject_replicates_from_regex(data, regex)

            delCols = []
            for subject in subjectDict:
                delCols.extend(subjectDict[subject])
                aux = data[subjectDict[subject]]
                data[subject] = calculate_median_replicates(aux, log)

            dataset = data.drop(delCols, 1)
            dataset = dataset.dropna(how='all')

    return dataset


def check_columns(data, req_columns, generated_columns):
    return set(req_columns).difference(set(data.columns)).difference(generated_columns)


def check_minimum_configuration(configuration):
    minimum_req = ['columns', 'indexCol',
                   'proteinCol', 'log',
                   'file', 'valueCol', 'attributes']

    return set(minimum_req).difference(set(configuration.keys()))


def load_dataset(uri, configuration):
    ''' This function gets the molecular data from a proteomics experiment.
        Input: uri of the processed file resulting from MQ
        Output: pandas DataFrame with the columns and filters defined in config.py '''
    data = None
    regexCols = None
    filters = None
    columns = configuration["columns"]
    regexCols = [c.replace("\\\\", "\\") for c in columns if '+' in c]
    columns = set(columns).difference(regexCols)
    generated_columns = []
    if 'generated_columns' in configuration:
        generated_columns = configuration['generated_columns']

    if 'filters' in configuration:
        filters = configuration["filters"]

    indexCol = configuration["indexCol"]
    data = builder_utils.readDataset(uri)
    missing_cols = check_columns(data, columns, generated_columns)
    if len(missing_cols) == 0:
        if filters is not None:
            data = data[data[filters].isnull().all(1)]
            data = data.drop(filters, axis=1)
            columns = set(columns).difference(filters)
        if 'numeric filter' in configuration:
            for f in configuration['numeric filter']:
                key = list(f.keys())[0]
                if key in columns:
                    value = f[key]
                    data = data[data[key] >= value]
                else:
                    raise Exception("Error when applying numeric filter on {}. The column is not in the dataset".format(f))
        data = data.dropna(subset=[configuration["proteinCol"]], axis=0)
        data = expand_groups(data, configuration)
        columns.remove(indexCol)

        for regex in regexCols:
            r = re.compile(regex)
            columns.update(set(filter(r.match, data.columns)))

        data = data[list(columns)].replace('Filtered', np.nan)
        value_cols = get_value_cols(data, configuration)
        data[value_cols] = data[value_cols].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        data = data.dropna(how='all', subset=value_cols, axis=0)
    else:
        raise Exception("Error when importing proteomics experiment.\n Missing columns: {}".format(",".join(missing_cols)))

    return data, regexCols


def remove_contaminant_tag(column, tag='CON__'):
    new_column = [c.replace(tag, '') for c in column]

    return new_column


def expand_groups(data, configuration):
    default_group_col = 'id'
    if "groupCol" not in configuration or configuration["groupCol"] is None:
        data.index.name = default_group_col
        data = data.reset_index()
        configuration['groupCol'] = default_group_col
    elif configuration['groupCol'] not in data.columns:
        data.index.name = configuration['groupCol']
        data = data.reset_index()

    s = data[configuration["proteinCol"]].str.split(';').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
    del data[configuration["proteinCol"]]
    pdf = s.to_frame(configuration["proteinCol"])
    if "multipositions" in configuration:
        s2 = data[configuration["multipositions"]].str.split(';').apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
        del data[configuration["multipositions"]]
        pdf = pd.concat([s, s2], axis=1, keys=[configuration["proteinCol"], configuration["multipositions"]])
    data = data.join(pdf)
    if 'contaminant_tag' in configuration:
        data[configuration["proteinCol"]] = remove_contaminant_tag(column=data[configuration["proteinCol"]].tolist(), tag=configuration['contaminant_tag'])
    data["is_razor"] = ~ data[configuration["groupCol"]].duplicated()
    data = data.set_index(configuration["indexCol"])

    return data

############## ProteinModification entity ####################


def extract_modification_protein_rels(data, configuration):
    modificationId = configuration["modId"]
    cols = configuration["positionCols"]
    aux = data[cols]
    aux = aux.reset_index()
    aux.columns = ["START_ID", "position", "residue"]
    aux["END_ID"] = modificationId
    aux['TYPE'] = "HAS_MODIFICATION"
    aux = aux[['START_ID', 'END_ID', 'TYPE', "position", "residue"]]
    aux['position'] = aux['position'].astype('int64')
    aux = aux.drop_duplicates()

    return aux


def extract_protein_modification_subject_rels(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    cols = [proteinCol]
    cols.extend(positionCols)
    data = data.reset_index()
    data["END_ID"] = data[proteinCol].map(str) + "_" + data[positionCols[1]].map(str) + data[positionCols[0]].map(str) + '-' + configuration["mod_acronym"]
    data = data.set_index("END_ID")
    newIndexdf = data.copy()
    data = data.drop(cols, axis=1)
    data = data.filter(regex=configuration["valueCol"].replace("\\\\", "\\"))
    data.columns = [c.split(" ")[1] for c in data.columns]
    data = data.stack()
    data = data.reset_index()
    data.columns = ["c"+str(i) for i in range(len(data.columns))]
    columns = ['END_ID', 'START_ID', "value"]
    attributes = configuration["attributes"]
    (cAttributes, cCols), (rAttributes, regexCols) = extract_attributes(newIndexdf, attributes)
    if not rAttributes.empty:
        data = merge_regex_attributes(data, rAttributes, ["c0", "c1"], regexCols)
        columns.extend(regexCols)
    if not cAttributes.empty:
        data = merge_col_attributes(data, cAttributes, "c0")
        columns.extend(cCols)

    data['TYPE'] = "HAS_QUANTIFIED_MODIFIED_PROTEIN"
    columns.append("TYPE")
    data.columns = columns
    data = data[['START_ID', 'END_ID', 'TYPE', "value"] + regexCols + cCols]
    data.columns = [c.replace('PG.', '') for c in data.columns]
    data = data.drop_duplicates()

    return data


def extract_protein_protein_modification_rels(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    cols = [proteinCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["END_ID"] = aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux = aux.drop(positionCols, axis=1)
    aux = aux.set_index("END_ID")
    aux = aux.reset_index()
    aux.columns = ["END_ID", "START_ID"]
    aux['TYPE'] = "HAS_MODIFIED_SITE"
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    aux = aux.drop_duplicates()

    return aux


def extract_peptide_protein_modification_rels(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    sequenceCol = configuration["sequenceCol"]
    cols = [sequenceCol, proteinCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["END_ID"] = aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux = aux.drop([proteinCol] + positionCols, axis=1)
    aux = aux.set_index("END_ID")
    aux = aux.reset_index()
    aux.columns = ["END_ID", "START_ID"]
    aux["START_ID"] = aux["START_ID"].str.upper()
    aux['TYPE'] = "HAS_MODIFIED_SITE"
    aux = aux[['START_ID', 'END_ID', 'TYPE']]
    aux = aux.drop_duplicates()

    return aux


def extract_protein_modifications_rels(data, configuration):
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    sequenceCol = configuration["sequenceCol"]
    cols = [proteinCol, sequenceCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["ID"] = aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux = aux.set_index("ID")
    aux = aux.reset_index()
    aux[sequenceCol] = aux[sequenceCol].str.replace('_', '-')
    aux["source"] = "experimentally_identified"
    aux.columns = ["ID", "protein", "sequence_window", "position", "residue", "source"]
    aux = aux.drop_duplicates()

    return aux


def extract_protein_modifications_modification_rels(data, configuration):
    modID = configuration["modId"]
    positionCols = configuration["positionCols"]
    proteinCol = configuration["proteinCol"]
    sequenceCol = configuration["sequenceCol"]
    cols = [proteinCol, sequenceCol]
    cols.extend(positionCols)
    aux = data.copy().reset_index()
    aux = aux[cols]
    aux["START_ID"] = aux[proteinCol].map(str) + "_" + aux[positionCols[1]].map(str) + aux[positionCols[0]].map(str)+'-'+configuration["mod_acronym"]
    aux["END_ID"] = modID
    aux = aux[["START_ID", "END_ID"]]

    return aux

################# Peptide entity ####################


def extract_peptides(data, configuration):
    aux = data.copy()
    modid = configuration["type"]
    aux["type"] = modid
    aux = aux["type"]
    aux = aux.reset_index()
    aux = aux.groupby(aux.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
    aux.columns = ["ID", "type", "count"]
    aux = aux.drop_duplicates()

    return aux


def extract_peptide_subject_rels(data, configuration):
    data = data[~data.index.duplicated(keep='first')]
    aux = data.filter(regex=configuration["valueCol"].replace("\\\\", "\\"))
    attributes = configuration["attributes"]
    aux.columns = [c.split(" ")[1] for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux.columns = ["c"+str(i) for i in range(len(aux.columns))]
    columns = ['END_ID', 'START_ID', "value"]

    (cAttributes, cCols), (rAttributes, regexCols) = extract_attributes(data, attributes)
    if not rAttributes.empty:
        aux = merge_regex_attributes(aux, rAttributes, ["c0", "c1"], regexCols)
        columns.extend(regexCols)
    if not cAttributes.empty:
        aux = merge_col_attributes(aux, cAttributes, "c0")
        columns.extend(cCols)

    aux['TYPE'] = "HAS_QUANTIFIED_PEPTIDE"
    columns.append("TYPE")
    aux.columns = columns
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"] + regexCols + cCols]
    aux.columns = [c.replace('PG.', '') for c in aux.columns]
    aux = aux.drop_duplicates()

    return aux


def extract_peptide_protein_rels(data, configuration):
    cols = [configuration["proteinCol"]]
    cols.extend(configuration["positionCols"])
    aux = data[cols]
    aux = aux.reset_index()
    aux.columns = ["Sequence", "Protein", "Start", "End"]
    aux['TYPE'] = "BELONGS_TO_PROTEIN"
    aux['source'] = 'experimentally_identified'
    aux.columns = ['START_ID', 'END_ID', "start", "end", 'TYPE', 'source']
    aux = aux[['START_ID', 'END_ID', 'TYPE', 'source']]
    return aux


def extract_protein_subject_rels(data, configuration):
    aux = data.filter(regex=configuration["valueCol"])
    attributes = configuration["attributes"]
    aux.columns = [re.sub("\.?" + configuration["valueCol"] + "\s?", '', c).strip() for c in aux.columns]
    aux = aux.stack()
    aux = aux.reset_index()
    aux.columns = ["c"+str(i) for i in range(len(aux.columns))]
    columns = ['END_ID', 'START_ID', "value"]
    (cAttributes, cCols), (rAttributes, regexCols) = extract_attributes(data, attributes)
    if not rAttributes.empty:
        aux = merge_regex_attributes(aux, rAttributes, ["c0", "c1"], regexCols)
        columns.extend(regexCols)
    if not cAttributes.empty:
        aux = merge_col_attributes(aux, cAttributes, "c0")
        columns.extend(cCols)
    aux['TYPE'] = "HAS_QUANTIFIED_PROTEIN"
    columns.append("TYPE")
    aux.columns = columns
    aux = aux[['START_ID', 'END_ID', 'TYPE', "value"] + regexCols + cCols]
    aux.columns = [c.replace('PG.', '') for c in aux.columns]

    return aux


def get_value_cols(data, configuration):
    value_cols = []
    if 'valueCol' in configuration:
        value_cols = [c for c in data.columns if configuration['valueCol'] in c]

    return value_cols


def extract_subject_replicates_from_regex(data, regex):
    subjectDict = defaultdict(list)
    for r in regex:
        columns = data.filter(regex=r).columns
        for c in columns:
            value = ""
            timepoint = ""
            fields = c.split('_')
            if len(fields) > 1:
                value = " ".join(fields[0].split(' ')[0:-1])
                subject = fields[1]
                if len(fields) > 2:
                    timepoint = " " + fields[2]
            else:
                subject = fields[0]
            ident = value + " " + subject + timepoint
            subjectDict[ident].append(c)

    return subjectDict


def extract_subject_replicates(data, value_cols):
    subjectDict = defaultdict(list)
    for c in value_cols:
        fields = c.split('_')
        value = " ".join(fields[0].split(' ')[0:-1])
        subject = fields[1]
        timepoint = ""
        if len(fields) > 2:
            timepoint = " " + fields[2]
        ident = value + " " + subject + timepoint
        subjectDict[ident].append(c)

    return subjectDict


def extract_attributes(data, attributes):
    auxAttr_col = pd.DataFrame(index=data.index)
    auxAttr_reg = pd.DataFrame(index=data.index)
    cCols = []
    regexCols = []
    for ctype in attributes:
        if ctype == "regex":
            for r in attributes[ctype]:
                attr_col = data.filter(regex=r)
                if not attr_col.empty:
                    regexCols.append(r)
                    auxAttr_reg = auxAttr_reg.join(attr_col)
        else:
            auxAttr_col = auxAttr_col.join(data[attributes[ctype]])
            cCols = [c.replace(' ', '_').replace('-', '') for c in attributes[ctype]]

    reg_attr_index = auxAttr_reg.index.name
    col_attr_index = auxAttr_col.index.name
    auxAttr_reg = auxAttr_reg.reset_index().drop_duplicates().set_index(reg_attr_index)
    auxAttr_col = auxAttr_col.reset_index().drop_duplicates().set_index(col_attr_index)

    return (auxAttr_col, cCols), (auxAttr_reg, regexCols)


def merge_regex_attributes(data, attributes, index, regexCols):
    data = data.sort_values(by=index)
    data = data.set_index(index)
    if not attributes.empty:
        for rc in regexCols:
            attr_aux = attributes.filter(regex=rc)
            columns = [re.sub("\.?" + rc + "\s?", '', c).strip() for c in attr_aux.columns]
            attr_aux.columns = columns
            attr_aux = attr_aux.stack()
            attr_aux = attr_aux.reset_index()
            attr_aux.columns = ["c" + str(i) for i in range(len(attr_aux.columns))]
            attr_aux = attr_aux.sort_values(by=index)
            data = data.join(attr_aux.set_index(index), rsuffix='test')
            del(attr_aux)
        del(attributes)
    data = data.reset_index()

    return data


def merge_col_attributes(data, attributes, index):
    if not attributes.empty:
        data = data.set_index(index)
        data = data.join(attributes)
        del(attributes)
        data = data.reset_index()

    return data


def calculate_median_replicates(data, log="log2"):
    median = None
    data = data.apply(pd.to_numeric, errors='coerce')
    if log == "log2":
        median = np.log2(data).replace([np.inf, -np.inf], np.nan).median(axis=1, skipna=True)
    elif log == "log10":
        median = np.log10(median).replace([np.inf, -np.inf], np.nan).median(axis=1, skipna=True)
    else:
        median = data.median(axis=1)

    return median


def update_groups(data, groups):
    data = data.join(groups.to_frame(), on='START_ID')

    return data


def get_dataset_configuration(processing_format, data_type):
    config = builder_utils.get_config(config_name="proteomics.yml", data_type='experiments')
    dataset_config = {}
    if processing_format in config:
        if data_type is not None:
            if data_type in config[processing_format]:
                dataset_config = config[processing_format][data_type]
        else:
            dataset_config = config[processing_format]

    return dataset_config
