import os
import json
from ckg.graphdb_builder import builder_utils
from ckg import ckg_utils
from ckg.analytics_core.analytics import analytics
from ckg.analytics_core.viz import viz
import pandas as pd
import itertools
import time

ckg_config = ckg_utils.read_ckg_config()
log_config = ckg_config['analytics_factory_log']
logger = builder_utils.setup_logging(log_config, key="analytics_factory")


class Analysis:
    def __init__(self, identifier, analysis_type, args, data, result=None, plots={}):
        self._identifier = identifier
        self._analysis_type = analysis_type
        self._args = args
        self._data = data
        self._result = result
        self._plots = plots
        if self._result is None:
            self._result = {}
            self.generate_result()

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        self._identifier = identifier

    @property
    def analysis_type(self):
        return self._analysis_type

    @analysis_type.setter
    def analysis_type(self, analysis_type):
        self._analysis_type = analysis_type

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = args

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        self._result = result

    @property
    def plots(self):
        return self._plots

    @plots.setter
    def plot(self, plots):
        self._plots = plots

    def update_plots(self, plots):
        self._plots.update(plots)

    def generate_result(self):
        logger.info("Generating result for: ", self.analysis_type)
        try:
            if self.analysis_type == "wide_format":
                r = analytics.transform_into_wide_format(self.data, self.args['index'], self.args['columns'], self.args['values'], extra=[self.args['extra']])
                self.result[self.analysis_type] = r
            elif self.analysis_type == "summary":
                r = analytics.get_summary_data_matrix(self.data)
                self.result[self.analysis_type] = r
            elif self.analysis_type == "normalization":
                method = 'median_polish'
                if 'method' in self.args:
                    method = self.args['method']
                self.result[self.analysis_type] = analytics.normalize_data(self.data, method=method)
            elif self.analysis_type == "pca":
                components = 2
                drop_cols = []
                group = 'group'
                annotation_cols = []
                if "components" in self.args:
                    components = self.args["components"]
                if "drop_cols" in self.args:
                    drop_cols = self.args["drop_cols"]
                if 'group' in self.args:
                    group = self.args['group']
                if 'hovering_cols' in self.args:
                    annotation_cols = self.args['hovering_cols']
                r, nargs = analytics.run_pca(self.data, components=components, drop_cols=drop_cols, group=group, annotation_cols=annotation_cols)
                self.result[self.analysis_type] = r
                self.args.update(nargs)
            elif self.analysis_type == 'functional_pca':
                dfid = 'processed'
                annotid = 'go annotation'
                key = 'nes'
                annotation_col = 'annotation'
                identifier_col = 'identifier'
                index = ['group', 'sample', 'subject']
                outdir = None
                min_size = 15
                max_size = 500
                scale = False
                permutations = 0
                components = 2
                drop_cols = []
                hovering_cols = []
                if 'data_id' in self.args:
                    dfid = self.args['data_id']
                if 'annotation_id' in self.args:
                    annotid = self.args['annotation_id']
                if 'key' in self.args:
                    key = self.args['key']
                if 'annotation_col' in self.args:
                    annotation_col = self.args['annotation_col']
                if 'identifier_col' in self.args:
                    identifier_col = self.args['identifier_col']
                if 'index' in self.args:
                    index = self.args['index']
                if 'outdir' in self.args:
                    outdir = self.args['outdir']
                if 'min_size' in self.args:
                    min_size = self.args['min_size']
                if 'max_size' in self.args:
                    max_size = args['max_size']
                if 'scale' in self.args:
                    scale = self.args['scale']
                if 'permutations' in self.args:
                    permutations = self.args['permutations']
                if "components" in self.args:
                    components = self.args["components"]
                if "drop_cols" in self.args:
                    drop_cols = self.args["drop_cols"]
                if 'hovering_cols' in self.args:
                    hovering_cols = self.args['hovering_cols']
                if dfid in self.data and annotid in self.data:
                    result = analytics.run_ssgsea(self.data[dfid], self.data[annotid], annotation_col=annotation_col,
                                                                        identifier_col=identifier_col, set_index=index, outdir=outdir, 
                                                                        min_size=min_size, max_size=max_size, scale=scale, permutations=permutations)
                    if key in result:
                        r, nargs = analytics.run_pca(result[key], components=components, drop_cols=drop_cols, annotation_cols=hovering_cols)
                        self.result[self.analysis_type] = r
                        self.args.update(nargs)
            elif self.analysis_type == "tsne":
                components = 2
                perplexity = 40
                n_iter = 1000
                drop_cols = []
                init = 'pca'
                annotation_cols = []
                if "components" in self.args:
                    components = self.args["components"]
                if "perplexity" in self.args:
                    perplexity = self.args["perplexity"]
                if "n_iter" in self.args:
                    n_iter = self.args["n_iter"]
                if "init" in self.args:
                    init = self.args["init"]
                if "drop_cols" in self.args:
                    drop_cols = self.args["drop_cols"]
                if 'hovering_cols' in self.args:
                    annotation_cols = self.args['hovering_cols']

                self.result, nargs = analytics.run_tsne(self.data, components=components, annotation_cols=annotation_cols, drop_cols=drop_cols, perplexity=perplexity, n_iter=n_iter, init=init)
                self.args.update(nargs)
            elif self.analysis_type == "umap":
                n_neighbors = 10
                min_dist = 0.3
                metric = 'cosine'
                annotation_cols = []
                if "n_neighbors" in self.args:
                    n_neighbors = self.args["n_neighbors"]
                if "min_dist" in self.args:
                    min_dist = self.args["min_dist"]
                if "metric" in self.args:
                    metric = self.args["metric"]
                if 'hovering_cols' in self.args:
                    annotation_cols = self.args['hovering_cols']
                if n_neighbors < self.data.shape[0]:
                    self.result, nargs = analytics.run_umap(self.data, n_neighbors=n_neighbors, annotation_cols=annotation_cols, min_dist=min_dist, metric=metric)
                    self.args.update(nargs)
            elif self.analysis_type == "mapper":
                n_cubes = 15
                overlap = 0.5
                n_clusters = 3
                linkage = "complete"
                affinity = "correlation"
                labels = {}
                if "labels" in self.args:
                    labels = self.args["labels"]
                if "n_cubes" in self.args:
                    n_cubes = self.args["n_cubes"]
                if "overlap" in self.args:
                    overlap = self.args["overlap"]
                if "n_clusters" in self.args:
                    n_clusters = self.args["n_clusters"]
                if "linkage" in self.args:
                    linkage = self.args["linkage"]
                if "affinity" in self.args:
                    affinity = self.args["affinity"]
                r, nargs = analytics.run_mapper(self.data, n_cubes=n_cubes, overlap=overlap, n_clusters=n_clusters, linkage=linkage, affinity=affinity)
                self.args.update(nargs)
                self.result[self.analysis_type] = r
            elif self.analysis_type == 'ttest':
                alpha = 0.05
                correction = 'fdr_bh'
                if "alpha" in self.args:
                    alpha = self.args["alpha"]
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                for pair in itertools.combinations(self.data.group.unique(), 2):
                    ttest_result = analytics.run_ttest(self.data, pair[0], pair[1], alpha=0.05, correction=correction)
                    self.result[pair] = ttest_result
            elif self.analysis_type == 'anova':
                start = time.time()
                alpha = 0.05
                drop_cols = []
                group = 'group'
                subject = 'subject'
                permutations = 50
                is_logged = True
                correction = 'fdr_bh'
                if "alpha" in self.args:
                    alpha = self.args["alpha"]
                if "drop_cols" in self.args:
                    drop_cols = self.args['drop_cols']
                if "subject" in self.args:
                    subject = self.args['subject']
                if "group" in self.args:
                    group = self.args["group"]
                if "permutations" in self.args:
                    permutations = self.args["permutations"]
                if "is_logged" in self.args:
                    is_logged = self.args['is_logged']
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                anova_result = analytics.run_anova(self.data, drop_cols=drop_cols, subject=subject, group=group, alpha=alpha, permutations=permutations, is_logged=is_logged, correction=correction)
                self.result[self.analysis_type] = anova_result
            elif self.analysis_type == 'ancova':
                alpha = 0.05
                drop_cols = []
                group = 'group'
                subject = 'subject'
                permutations = 50
                is_logged = True
                correction = 'fdr_bh'
                covariates = []
                if "alpha" in self.args:
                    alpha = self.args["alpha"]
                if "drop_cols" in self.args:
                    drop_cols = self.args['drop_cols']
                if "subject" in self.args:
                    subject = self.args['subject']
                if "group" in self.args:
                    group = self.args["group"]
                if "permutations" in self.args:
                    permutations = self.args["permutations"]
                if "is_logged" in self.args:
                    is_logged = self.args['is_logged']
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                if "covariates" in self.args:
                    covariates = self.args["covariates"]
                if 'processed' in self.data and 'metadata' in self.data:
                    metadata = self.data['metadata']
                    processed = self.data['processed']
                    df = processed.set_index([subject, group]).join(metadata.set_index([subject, group])[covariates]).reset_index()
                    ancova_result = analytics.run_ancova(df, covariates=covariates, drop_cols=drop_cols, subject=subject, group=group, alpha=alpha, permutations=permutations, is_logged=is_logged, correction=correction)
                    self.result[self.analysis_type] = ancova_result
            elif self.analysis_type == 'qcmarkers':
                sample_col = 'sample'
                group_col = 'group'
                identifier_col = 'identifier'
                qcidentifier_col = 'identifier'
                qcclass_col = 'class'
                drop_cols = ['subject']
                if 'drop_cols' in self.args:
                    drop_cols = self.args['drop_cols']
                if 'sample_col' in self.args:
                    sample_col = self.args['sample_col']
                if 'group_col' in self.args:
                    group_col = self.args['group_col']
                if 'identifier_col' in self.args:
                    identifier_col = self.args['identifier_col']
                if 'qcidentifier_col' in self.args:
                    qcidentifier_col = self.args['qcidentifier_col']
                if 'qcclass_col' in self.args:
                    qcclass_col = self.args['qcclass_col']
                if 'processed' in self.data and 'tissue qcmarkers' in self.data:
                    processed_data = self.data['processed']
                    qcmarkers = self.data['tissue qcmarkers']
                    self.result[self.analysis_type] = analytics.run_qc_markers_analysis(processed_data, qcmarkers, sample_col, group_col, drop_cols, identifier_col, qcidentifier_col, qcclass_col)
            elif self.analysis_type == 'samr':
                start = time.time()
                alpha = 0.05
                s0 = None
                drop_cols = []
                group = 'group'
                subject = 'subject'
                permutations = 250
                fc = 0
                is_logged = True
                if "alpha" in self.args:
                    alpha = self.args["alpha"]
                if "drop_cols" in self.args:
                    drop_cols = self.args['drop_cols']
                if "subject" in self.args:
                    subject = self.args['subject']
                if "group" in self.args:
                    group = self.args["group"]
                if "s0" in self.args:
                    s0 = self.args["s0"]
                if "permutations" in self.args:
                    permutations = self.args["permutations"]
                if "fc" in self.args:
                    fc = self.args['fc']
                if "is_logged" in self.args:
                    is_logged = self.args['is_logged']
                anova_result = analytics.run_samr(self.data, drop_cols=drop_cols, subject=subject, group=group, alpha=alpha, s0=s0, permutations=permutations, fc=fc, is_logged=is_logged)
                self.result[self.analysis_type] = anova_result
            elif self.analysis_type == '2-way anova':
                drop_cols = []
                subject = 'subject'
                group = ['group', 'secondary_group']
                if "drop_cols" in self.args:
                    drop_cols = self.args['drop_cols']
                if "subject" in self.args:
                    subject = self.args["subject"]
                if "group" in self.args:
                    group = self.args["group"]
                two_way_anova_result = analytics.run_two_way_anova(self.data, drop_cols=drop_cols, subject=subject, group=group)
                self.result[self.analysis_type] = two_way_anova_result
            elif self.analysis_type == "repeated_measurements_anova":
                start = time.time()
                alpha = 0.05
                drop_cols = []
                within = 'group'
                subject = 'subject'
                permutations = 50
                correction = 'fdr_bh'
                if "alpha" in self.args:
                    alpha = self.args["alpha"]
                if "drop_cols" in self.args:
                    drop_cols = self.args['drop_cols']
                if "group" in self.args:
                    within = self.args["within"]
                if "subject" in self.args:
                    subject = self.args["subject"]
                if "permutations" in self.args:
                    permutations = self.args["permutations"]
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                anova_result = analytics.run_repeated_measurements_anova(self.data, drop_cols=drop_cols, subject=subject, within=within, alpha=alpha, permutations=permutations, correction=correction)
                self.result[self.analysis_type] = anova_result
            elif self.analysis_type == "mixed_anova":
                start = time.time()
                alpha = 0.05
                drop_cols = []
                within = 'group'
                between = 'group2'
                subject = 'subject'
                permutations = 50
                correction = 'fdr_bh'
                is_logged = True
                if "alpha" in self.args:
                    alpha = self.args["alpha"]
                if "drop_cols" in self.args:
                    drop_cols = self.args['drop_cols']
                if "within" in self.args:
                    within = self.args["within"]
                if "subject" in self.args:
                    subject = self.args["subject"]
                if "permutations" in self.args:
                    permutations = self.args["permutations"]
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                if "between" in self.args:
                    between = self.args["between"]
                if 'is_logged' in self.args:
                    is_logged = self.args['is_logged']
                if 'processed' in self.data and 'metadata' in self.data:
                    metadata = self.data['metadata']
                    processed = self.data['processed']
                    df = processed.set_index([subject, within]).join(metadata.set_index([subject, within])[between]).reset_index()
                    anova_result = analytics.run_mixed_anova(df, between=between, drop_cols=drop_cols, subject=subject, within=within, alpha=alpha, permutations=permutations, is_logged=is_logged, correction=correction)
                    self.result[self.analysis_type] = anova_result
            elif self.analysis_type == "dabest":
                drop_cols = []
                group = 'group'
                subject = 'subject'
                test = 'mean_diff'
                if "drop_cols" in self.args:
                    drop_cols = self.args['drop_cols']
                if "group" in self.args:
                    group = self.args["group"]
                if "subject" in self.args:
                    subject = self.args["subject"]
                if "test" in self.args:
                    test = self.args["test"]
                dabest_result = analytics.run_dabest(self.data, drop_cols=drop_cols, subject=subject, group=group, test=test)
                self.result[self.analysis_type] = dabest_result
            elif self.analysis_type == "correlation":
                start = time.time()
                alpha = 0.05
                method = 'pearson'
                correction = 'fdr_bh'
                subject = 'subject'
                group = 'group'
                if 'group' in self.args:
                    group = self.args['group']
                if 'subject' in self.args:
                    subject = self.args['subject']
                if "alpha" in self.args:
                    alpha = self.args["args"]
                if "method" in self.args:
                    method = self.args["method"]
                if "correction" in self.args:
                    correction = self.args["correction"]
                self.result[self.analysis_type] = analytics.run_correlation(self.data, alpha=alpha, subject=subject, group=group, method=method, correction=correction)
            elif self.analysis_type == "repeated_measurements_correlation":
                start = time.time()
                alpha = 0.05
                method = 'pearson'
                correction = 'fdr_bh'
                cutoff = 0.5
                subject = 'subject'
                if 'subject' in self.args:
                    subject = self.args['subject']
                if "alpha" in self.args:
                    alpha = self.args["args"]
                if "method" in self.args:
                    method = self.args["method"]
                if "correction" in self.args:
                    correction = self.args["correction"]
                self.result[self.analysis_type] = analytics.run_rm_correlation(self.data, alpha=alpha, subject=subject, correction=correction)
            elif self.analysis_type == "merge_for_polar":
                theta_col = 'modifier'
                group_col = 'group'
                identifier_col = 'identifier'
                normalize = True
                aggr_func = 'mean'
                if 'group_col' in self.args:
                    group_col = self.args['group_col']
                if 'theta_col' in self.args:
                    theta_col = self.args['theta_col']
                if 'identifier_col' in self.args:
                    identifier_col = self.args['identifier_col']
                if 'aggr_func' in self.args:
                    aggr_func = self.args['aggr_func']
                if 'normalize' in self.args:
                    normalize = self.args['normalize']
                if 'regulation_data' in self.args and 'regulators' in self.args:
                    if self.args['regulation_data'] in self.data and self.args['regulators'] in self.data:
                        self.result[self.analysis_type] = analytics.merge_for_polar(self.data[self.args['regulation_data']], self.data[self.args['regulators']], identifier_col=identifier_col, group_col=group_col, theta_col=theta_col, aggr_func=aggr_func, normalize=normalize)
            elif self.analysis_type == "regulation_enrichment":
                start = time.time()
                identifier = 'identifier'
                groups = ['group1', 'group2']
                annotation_col = 'annotation'
                reject_col = 'rejected'
                method = 'fisher'
                annotation_type = 'functional'
                correction = 'fdr_bh'
                if 'identifier' in self.args:
                    identifier = self.args['identifier']
                if 'groups' in self.args:
                    groups = self.args['groups']
                if 'annotation_col' in self.args:
                    annotation_col = self.args['annotation_col']
                if 'reject_col' in self.args:
                    reject_col = self.args['reject_col']
                if 'method' in self.args:
                    method = self.args['method']
                if 'annotation_type' in self.args:
                    annotation_type = self.args['annotation_type']
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                if 'regulation_data' in self.args and 'annotation' in self.args:
                    if self.args['regulation_data'] in self.data and self.args['annotation'] in self.data:
                        self.analysis_type = annotation_type+"_"+self.analysis_type
                        self.result[self.analysis_type] = analytics.run_regulation_enrichment(self.data[self.args['regulation_data']], self.data[self.args['annotation']],
                                                                                            identifier=identifier, groups=groups, annotation_col=annotation_col, reject_col=reject_col,
                                                                                            method=method, correction=correction)
                print('Enrichment', time.time() - start)
            elif self.analysis_type == "up_down_enrichment":
                start = time.time()
                identifier = 'identifier'
                groups = ['group1', 'group2']
                annotation_col = 'annotation'
                reject_col = 'rejected'
                method = 'fisher'
                annotation_type = 'functional'
                correction = 'fdr_bh'
                alpha = 0.05
                lfc_cutoff = 1
                if 'identifier' in self.args:
                    identifier = self.args['identifier']
                if 'groups' in self.args:
                    groups = self.args['groups']
                if 'annotation_col' in self.args:
                    annotation_col = self.args['annotation_col']
                if 'reject_col' in self.args:
                    reject_col = self.args['reject_col']
                if 'method' in self.args:
                    method = self.args['method']
                if 'annotation_type' in self.args:
                    annotation_type = self.args['annotation_type']
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                if 'alpha' in self.args:
                    alpha = self.args['alpha']
                if 'lfc_cutoff' in self.args:
                    lfc_cutoff = self.args['lfc_cutoff']
                if 'regulation_data' in self.args and 'annotation' in self.args:
                    if self.args['regulation_data'] in self.data and self.args['annotation'] in self.data:
                        self.analysis_type = annotation_type+"_"+self.analysis_type
                        self.result[self.analysis_type] = analytics.run_up_down_regulation_enrichment(self.data[self.args['regulation_data']], self.data[self.args['annotation']],
                                                                                            identifier=identifier, groups=groups, annotation_col=annotation_col, reject_col=reject_col,
                                                                                            method=method, correction=correction, alpha=alpha, lfc_cutoff=lfc_cutoff)
                print('Enrichment', time.time() - start)
            elif self.analysis_type == "regulation_site_enrichment":
                start = time.time()
                identifier = 'identifier'
                groups = ['group1', 'group2']
                annotation_col = 'annotation'
                reject_col = 'rejected'
                method = 'fisher'
                annotation_type = 'functional'
                regex = r"(\w+~.+)_\w\d+\-\w+"
                correction = 'fdr_bh'
                if 'identifier' in self.args:
                    identifier = self.args['identifier']
                if 'groups' in self.args:
                    groups = self.args['groups']
                if 'annotation_col' in self.args:
                    annotation_col = self.args['annotation_col']
                if 'reject_col' in self.args:
                    reject_col = self.args['reject_col']
                if 'method' in self.args:
                    method = self.args['method']
                if 'annotation_type' in self.args:
                    annotation_type = self.args['annotation_type']
                if 'regex' in self.args:
                    regex = self.args['regex']
                if 'correction_method' in self.args:
                    correction = self.args['correction_method']
                if 'regulation_data' in self.args and 'annotation' in self.args:
                    if self.args['regulation_data'] in self.data and self.args['annotation'] in self.data:
                        self.analysis_type = annotation_type+"_"+self.analysis_type
                        self.result[self.analysis_type] = analytics.run_site_regulation_enrichment(self.data[self.args['regulation_data']],
                                                                                                self.data[self.args['annotation']], identifier=identifier,
                                                                                                groups=groups, annotation_col=annotation_col, reject_col=reject_col,
                                                                                                method=method, regex=regex, correction=correction)
            elif self.analysis_type == 'ssgsea':
                dfid = 'processed'
                annotid = 'go annotation'
                annotation_col = 'annotation'
                identifier_col = 'identifier'
                index = ['group', 'sample','subject']
                outdir = None
                min_size = 15
                scale = False
                permutations = 0
                if 'data_id' in self.args:
                    dfid = self.args['data_id']
                if 'annotation_id' in self.args:
                    annotid = self.args['annotation_id']
                if 'annotation_col' in self.args:
                    annotation_col = self.args['annotation_col']
                if 'identifier_col' in self.args:
                    identifier_col = self.args['identifier_col']
                if 'index' in self.args:
                    index = self.args['index']
                if 'outdir' in self.args:
                    outdir = self.args['outdir']
                if 'min_size' in self.args:
                    min_size = self.args['min_size']
                if 'scale' in self.args:
                    scale = self.args['scale']
                if 'permutations' in self.args:
                    permutations = self.args['permutations']

                if dfid in self.data and annotid in self.data:
                    self.result[self.analysis_type] = analytics.run_ssgsea(self.data[dfid], self.data[annotid], annotation_col=annotation_col,
                                                                        identifier_col=identifier_col, set_index=index, outdir=outdir,
                                                                        min_size=min_size, scale=scale, permutations=permutations)
            elif self.analysis_type == 'long_format':
                self.result[self.analysis_type] = analytics.transform_into_long_format(self.data, drop_columns=self.args['drop_columns'], group=self.args['group'], columns=self.args['columns'])
            elif self.analysis_type == 'ranking_with_markers':
                start = time.time()
                list_markers = []
                annotations = {}
                marker_col = 'identifier'
                marker_of_col = 'disease'
                if 'identifier' in self.args:
                    marker_col = self.args['identifier']
                if 'marker_of' in self.args:
                    marker_of_col = self.args['marker_of']
                if 'markers' in self.args:
                    if self.args['markers'] in self.data:
                        if marker_col in self.data[self.args['markers']]:
                            list_markers = self.data[self.args['markers']][marker_col].tolist()
                            if 'annotate' in self.args:
                                if self.args['annotate']:
                                    annotations = pd.Series(self.data[self.args['markers']][marker_of_col].values, index=self.data[self.args['markers']][marker_col]).to_dict()
                self.args['annotations'] = annotations
                if 'data' in self.args:
                    if self.args['data'] in self.data:
                        self.result[self.analysis_type] = analytics.get_ranking_with_markers(self.data[self.args['data']], drop_columns=self.args['drop_columns'], group=self.args['group'], columns=self.args['columns'], list_markers=list_markers, annotation = annotations)
            elif self.analysis_type == 'coefficient_of_variation':
                self.result[self.analysis_type] = analytics.get_coefficient_variation(self.data, drop_columns=self.args['drop_columns'], group=self.args['group'], columns=self.args['columns'])
            elif self.analysis_type == 'publications_abstracts':
                self.result[self.analysis_type] = analytics.get_publications_abstracts(self.data, publication_col="publication", join_by=['publication', 'Proteins', 'Diseases'], index="PMID")
            elif self.analysis_type == "wgcna":
                start = time.time()
                drop_cols_exp = []
                drop_cols_cli = []
                RsquaredCut = 0.8
                networkType = 'unsigned'
                minModuleSize = 30
                deepSplit = 2
                pamRespectsDendro = False
                merge_modules = True
                MEDissThres = 0.25
                verbose = 0
                sd_cutoff = 0
                if "drop_cols_exp" in self.args:
                    drop_cols_exp = self.args['drop_cols_exp']
                if "drop_cols_cli" in self.args:
                    drop_cols_cli = self.args['drop_cols_cli']
                if "RsquaredCut" in self.args:
                    RsquaredCut = self.args["RsquaredCut"]
                if "networkType" in self.args:
                    networkType = self.args["networkType"]
                if "minModuleSize" in self.args:
                    minModuleSize = self.args["minModuleSize"]
                if "deepSplit" in self.args:
                    deepSplit = self.args["deepSplit"]
                if "pamRespectsDendro" in self.args:
                    pamRespectsDendro = self.args["pamRespectsDendro"]
                if "merge_modules" in self.args:
                    merge_modules = self.args["merge_modules"]
                if "MEDissThres" in self.args:
                    MEDissThres = self.args["MEDissThres"]
                if "verbose" in self.args:
                    verbose = self.args["verbose"]
                if "sd_cutoff" in self.args:
                    sd_cutoff = self.args["sd_cutoff"]
                self.result[self.analysis_type] = analytics.run_WGCNA(self.data, drop_cols_exp, drop_cols_cli, RsquaredCut=RsquaredCut, networkType=networkType,
                                                                minModuleSize=minModuleSize, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro, merge_modules=merge_modules,
                                                                MEDissThres=MEDissThres, verbose=verbose, sd_cutoff=sd_cutoff)
            elif self.analysis_type == 'kaplan_meier':
                time_col = None
                event_col = None
                group_col = 'group'
                if 'time_col' in self.args:
                    time_col = self.args['time_col']
                if 'event_col' in self.args:
                    event_col = self.args['event_col']
                if 'group_col' in self.args:
                    group_col = self.args['group_col']
                self.result[self.analysis_type] = analytics.run_km(self.data, time_col, event_col, group_col, self.args)
            elif self.analysis_type == 'multi_correlation':
                start = time.time()
                alpha = 0.05
                method = 'pearson'
                correction = 'fdr_bh'
                subject = 'subject'
                group = 'group'
                on = ['subject', 'group']
                if 'on_cols' in self.args:
                    on = self.args['on_cols']
                if 'group' in self.args:
                    group = self.args['group']
                if 'subject' in self.args:
                    subject = self.args['subject']
                if "alpha" in self.args:
                    alpha = self.args["args"]
                if "method" in self.args:
                    method = self.args["method"]
                if "correction" in self.args:
                    correction = self.args["correction"]
                self.result[self.analysis_type] = analytics.run_multi_correlation(self.data, alpha=alpha, subject=subject, group=group, on=on, method=method, correction=correction)
            logger.info("Done with analysis: ", self.analysis_type)
        except Exception as e:
            logger.error("Error in analysis {}: {}".format(self.analysis_type, e))

    def get_plot(self, name, identifier):
        plot = []
        if len(self.result) >= 1:
            if name == "basicTable":
                colors = ('#C2D4FF', '#F5F8FF')
                columns = None
                rows = None
                figure_title = 'Basic table'
                if "colors" in self.args:
                    colors = self.args["colors"]
                if "cols" in self.args:
                    columns = self.args["cols"]
                if "rows" in self.args:
                    rows = self.args["rows"]
                if "title" in self.args:
                    figure_title = self.args["title"]
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args["title"] + id[0]+" vs "+id[1]
                    if isinstance(self.result[id], dict):
                        i = 0
                        for ident in self.result[id]:
                            plot.append(viz.get_table(self.result[id][ident], identifier+"_"+str(i), args={'title': figure_title+" "+ident, 'colors': colors, 'cols': columns, 'rows': rows,'width': 800, 'height': 1500, 'font': 12}))
                            i += 1
                    else:
                        plot.append(viz.get_table(self.result[id], identifier, args={'title': figure_title, 'colors': colors, 'cols': columns, 'rows': rows,'width': 800, 'height': 1500, 'font': 12}))
            if name == "multiTable":
                for id in self.result:
                    plot.append(viz.get_multi_table(self.result[id], identifier, self.args["title"]))
            elif name == "barplot":
                x_title = "x"
                y_title = "y"
                if "x_title" in self.args:
                    x_title = self.args["x_title"]
                if "y_title" in self.args:
                    y_title = self.args["y_title"]
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args["title"] = figure_title
                    plot.append(viz.get_barplot(self.result[id], identifier, self.args))
            elif name == "facetplot":
                x_title = "x"
                y_title = "y"
                plot_type = "bar"
                if "x_title" not in self.args:
                    self.args["x_title"] = x_title
                if "y_title" not in self.args:
                    self.args["y_title"] = y_title
                if "plot_type" not in self.args:
                    self.args["plot_type"] = plot_type
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args['title'] = figure_title
                    plot.append(viz.get_facet_grid_plot(self.result[id], identifier, self.args))
            elif name == "scatterplot":
                x_title = "x"
                y_title = "y"
                if "x_title" in self.args:
                    x_title = self.args["x_title"]
                if "y_title" in self.args:
                    y_title = self.args["y_title"]
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args['title'] = figure_title
                    plot.append(viz.get_scatterplot(self.result[id], identifier, self.args))
            elif name == 'pca':
                x_title = "x"
                y_title = "y"
                if "x_title" in self.args:
                    x_title = self.args["x_title"]
                if "y_title" in self.args:
                    y_title = self.args["y_title"]
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args['title'] = figure_title
                    plot.append(viz.get_pca_plot(self.result[id], identifier, self.args))
            elif name == "volcanoplot":
                alpha = 0.05
                lfc = 1.0
                if "alpha" not in self.args:
                    self.args["alpha"] = alpha
                if "lfc" not in self.args:
                    self.args["lfc"] = lfc
                for pair in self.result:
                    signature = self.result[pair]
                    self.args["title"] = self.args['title'] + " " + pair[0] + " vs " + pair[1]
                    p = viz.run_volcano(signature, identifier + "_" + pair[0] + "_vs_" + pair[1], self.args)
                    plot.extend(p)
            elif name == "enrichment_plot":
                for pair in self.result:
                    plots = viz.get_enrichment_plots(self.result[pair], identifier=pair, args=self.args)
                plot.extend(plots)
            elif name == 'network':
                source = 'source'
                target = 'target'
                if "source" not in self.args:
                    self.args["source"] = source
                if "target" not in self.args:
                    self.args["target"] = target
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args["title"] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args["title"]
                    self.args["title"] = figure_title
                    plot.append(viz.get_network(self.result[id], identifier, self.args))
            elif name == "heatmap":
                for id in self.result:
                    if not self.result[id].empty:
                        if isinstance(id, tuple):
                            identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                            figure_title = self.args["title"] + id[0]+" vs "+id[1]
                        else:
                            figure_title = self.args["title"]
                        self.args["title"] = figure_title
                        plot.append(viz.get_complex_heatmapplot(self.result[id], identifier, self.args))
            elif name == "mapper":
                for id in self.result:
                    labels = {}
                    if "labels" not in self.args:
                        self.args["labels"] = labels
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    plot.append(viz.getMapperFigure(self.result[id], identifier, title=figure_title, labels=self.args["labels"]))
            elif name == "scatterplot_matrix":
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args["title"] = figure_title
                    plot.append(viz.get_scatterplot_matrix(self.result[id], identifier, self.args))
            elif name == "distplot":
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args["title"] = figure_title
                    plot.extend(viz.get_distplot(self.result[id], identifier, self.args))
            elif name == "violinplot":
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args["title"] = figure_title
                    plot.extend(viz.get_violinplot(self.result[id], identifier, self.args))
            elif name == "polar":
                for id in self.result:
                    figure_title = self.args['title']
                    plot.append(viz.get_polar_plot(self.result[id], identifier, self.args))
            elif name == "km":
                for id in self.result:
                    plot.append(viz.get_km_plot(self.result[id], identifier, self.args))
            elif name == "wgcnaplots":
                start = time.time()
                data = {}
                wgcna_data = self.result
                if 'drop_cols_exp' in self.args and 'drop_cols_cli' in self.args:
                    if 'wgcna' in wgcna_data and wgcna_data['wgcna'] is not None:
                        for dtype in wgcna_data['wgcna']:
                            data = wgcna_data['wgcna'][dtype]
                            plot.extend(viz.get_WGCNAPlots(data, identifier + "-" + dtype))
                print('WGCNA-plot', time.time() - start)
            elif name == 'ranking':
                for id in self.result:
                    plot.append(viz.get_ranking_plot(self.result[id], identifier, self.args))
            elif name == 'qcmarkers_boxplot':
                for id in self.result:
                    plot.append(viz.get_boxplot_grid(self.result[id], identifier, self.args))
            elif name == 'clustergrammer':
                for id in self.result:
                    plot.append(viz.get_clustergrammer_plot(self.result[id], identifier, self.args))
            elif name == 'cytonet':
                for id in self.result:
                    plot.append(viz.get_cytoscape_network(self.result[id], identifier, self.args))
            elif name == 'wordcloud':
                for id in self.result:
                    plot.append(viz.get_wordcloud(self.result[id], identifier, self.args))

        self.update_plots({identifier: plot})

        return plot

    def publish_analysis(self, directory):
        builder_utils.checkDirectory(directory)
        plots_directory = os.path.join(directory, 'figures')
        results_directory = os.path.join(directory, 'results')
        builder_utils.checkDirectory(plots_directory)
        builder_utils.checkDirectory(results_directory)
        self.save_analysis_plots(plots_directory)
        self.save_analysis_result(results_directory)

    def save_analysis_result(self, results_directory):
        if self.result is not None:
            for analysis_type in self.result:
                result_json = {'args': self.args}
                result_str = ''
                if isinstance(self.result[analysis_type], dict):
                    for key in self.result[analysis_type]:
                        if isinstance(self.result[analysis_type][key], pd.DataFrame):
                            result_str[key] = self.result[analysis_type][key].to_json()
                elif isinstance(self.result[analysis_type], list) or isinstance(self.result[analysis_type], tuple):
                    result_str = []
                    for res in self.result[analysis_type]:
                        result_str.append(res.to_json())
                else:
                    result_str = self.result[analysis_type].to_json()

                result_json.update({'result': result_str})

                with open(os.path.join(results_directory, self.identifier+'_'+analysis_type+'.json'), 'w') as rf:
                    rf.write(json.dumps(result_json))

    def save_analysis_plots(self, plots_directory):
        for figure_id in self.plots:
            plot_format = 'json'
            plot = self.plots[figure_id]
            if isinstance(plot, dict):
                figure_json = {}
                if 'net_json' in plot:
                    figure_json['net_json'] = plot['net_json']
                if 'notebook' in plot:
                    figure_json['notebook'] = plot['notebook']
                if 'app' in plot:
                    json_str = ckg_utils.convert_dash_to_json(plot['app'])
                    figure_json['app'] = json_str
                if 'net_tables' in plot:
                    json_str_nodes = ckg_utils.convert_dash_to_json(plot['net_tables'][0])
                    json_str_edges = ckg_utils.convert_dash_to_json(plot['net_tables'][1])
                    figure_json["net_tables"] = (json_str_nodes, json_str_edges)
                figure_json = json.dumps(figure_json, cls=ckg_utils.NumpyEncoder)
            elif isinstance(plot, list):
                json_items = []
                for p in plot:
                    json_items.append(ckg_utils.convert_dash_to_json(p))
                figure_json = json.dumps(json_items, cls=ckg_utils.NumpyEncoder)
            elif isinstance(plot, str):
                figure_json = plot
                plot_format = 'html'
            else:
                json_str = ckg_utils.convert_dash_to_json(plot)
                figure_json = json.dumps(json_str, cls=ckg_utils.NumpyEncoder)

            with open(os.path.join(plots_directory, figure_id+'.'+plot_format), 'w') as ff:
                ff.write(figure_json)

    def make_interactive(self, name, identifier):
        if name == "volcanoplot":
            pass
