from report_manager.analyses import wgcnaAnalysis as wgcna
import report_manager.analyses.basicAnalysis as analyses
from report_manager.plots import basicFigures as figure
import pandas as pd
import itertools
import time

class AnalysisResult:
    def __init__(self, identifier, analysis_type, args, data, result=None):
        self._identifier = identifier
        self._analysis_type = analysis_type
        self._args = args
        self._data = data
        self._result = result
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

    def generate_result(self):
        if self.analysis_type == "wide_format":
            r = analyses.transform_into_wide_format(self.data, self.args['index'], self.args['x'], self.args['y'], extra=[self.args['group']])
            self.result[self.analysis_type] = r
        if self.analysis_type == "pca":
            components = 2
            if "components" in self.args:
                components = self.args["components"]
            self.result, nargs = analyses.run_pca(self.data, components=components)
            self.args.update(nargs)
        elif self.analysis_type  == "tsne":
            components = 2
            perplexity = 40
            n_iter = 1000
            init='pca'
            if "components" in self.args:
                components = self.args["components"]
            if "perplexity" in self.args:
                perplexity = self.args["perplexity"]
            if "n_iter" in self.args:
                n_iter = self.args["n_iter"]
            if "init" in self.args:
                init = self.args["init"]
            self.result, nargs = analyses.run_tsne(self.data, components=components, perplexity=perplexity, n_iter=n_iter, init=init)
            self.args.update(nargs)
        elif self.analysis_type  == "umap":
            n_neighbors=10
            min_dist=0.3
            metric='cosine'
            if "n_neighbors" in self.args:
                n_neighbors = self.args["n_neighbors"]
            if "min_dist" in self.args:
                min_dist = self.args["min_dist"]
            if "metric" in self.args:
                metric = self.args["metric"]
            if n_neighbors < self.data.shape[0]:
                self.result, nargs = analyses.run_umap(self.data, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
                self.args.update(nargs)
        elif self.analysis_type  == "mapper":
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
            r, nargs = analyses.run_mapper(self.data, n_cubes=n_cubes, overlap=overlap, n_clusters=n_clusters, linkage=linkage, affinity=affinity)
            self.args.update(nargs)
            self.result[self.analysis_type] = r
        elif self.analysis_type  == 'ttest':
            alpha = 0.05
            if "alpha" in self.args:
                alpha = self.args["alpha"]
            for pair in itertools.combinations(self.data.group.unique(),2):
                ttest_result = analyses.run_ttest(self.data, pair[0], pair[1], alpha = 0.05)
                self.result[pair] = ttest_result
        elif self.analysis_type  == 'anova':
            start = time.time()
            alpha = 0.05
            drop_cols = []
            group = 'group'
            permutations = 150
            if "alpha" in self.args:
                alpha = self.args["alpha"]
            if "drop_cols" in self.args:
                drop_cols = self.args['drop_cols']
            if "group" in self.args:
                group = self.args["group"]
            if "permutations" in self.args:
                permutations = self.args["permutations"]
            anova_result = analyses.run_anova(self.data, drop_cols=drop_cols, group=group, alpha=alpha, permutations=permutations)
            self.result[self.analysis_type] = anova_result
            print('ANOVA', time.time() - start)
        elif self.analysis_type  == '2-way anova':
            alpha = 0.05
            drop_cols = []
            variables = ['factor A', 'factor B']
            subject = 'subject'
            if "alpha" in self.args:
                alpha = self.args["alpha"]
            if "drop_cols" in self.args:
                drop_cols = self.args['drop_cols']
            if "subject" in self.args:
                subject = self.args["subject"]
            if "variables" in self.args:
                variables = self.args["variables"]
            two_way_anova_result = analyses.run_two_way_anova(self.data, variables=variables, drop_cols=drop_cols, subject=subject, alpha=alpha)
            self.result[self.analysis_type] = two_way_anova_result
        elif self.analysis_type == "repeated_measurements_anova":
            start = time.time()
            alpha = 0.05
            drop_cols = []
            group = 'group'
            subject = 'subject'
            permutations = 150
            if "alpha" in self.args:
                alpha = self.args["alpha"]
            if "drop_cols" in self.args:
                drop_cols = self.args['drop_cols']
            if "group" in self.args:
                group = self.args["group"]
            if "subject" in self.args:
                subject = self.args["subject"]
            if "permutations" in self.args:
                permutations = self.args["permutations"]
            anova_result = analyses.run_repeated_measurements_anova(self.data, drop_cols=drop_cols, subject=subject, group=group, alpha=alpha, permutations=permutations)
            self.result[self.analysis_type] = anova_result
            print('repeated-ANOVA', time.time() - start)
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
            dabest_result = analyses.run_dabest(self.data, drop_cols=drop_cols, subject=subject, group=group, test=test)
            self.result[self.analysis_type] = dabest_result
        elif self.analysis_type  == "correlation":
            start = time.time()
            alpha = 0.05
            method = 'pearson'
            correction = ('fdr', 'indep')
            subject='subject'
            group='group'
            if 'group' in self.args:
                group = self.args['group']
            if 'subject' in self.args:
                subject= self.args['subject']
            if "alpha" in self.args:
                alpha = self.args["args"]
            if "method" in self.args:
                method = self.args["method"]
            if "correction" in self.args:
                correction = self.args["correction"]
            self.result[self.analysis_type] = analyses.run_correlation(self.data, alpha=alpha, subject=subject, group=group, method=method, correction=correction)
            print('Correlation', time.time() - start)
        elif self.analysis_type  == "repeated_measurements_correlation":
            start = time.time()
            alpha = 0.05
            method = 'pearson'
            correction = ('fdr', 'indep')
            cutoff = 0.5
            subject='subject'
            if 'subject' in self.args:
                subject= self.args['subject']
            if "alpha" in self.args:
                alpha = self.args["args"]
            if "method" in self.args:
                method = self.args["method"]
            if "correction" in self.args:
                correction = self.args["correction"]
            if "cutoff" in self.args:
                cutoff = self.args['cutoff']
            self.result[self.analysis_type] = analyses.run_rm_correlation(self.data, alpha=alpha, subject=subject, correction=correction)
            print('repeated-Correlation', time.time() - start)
        elif self.analysis_type == "regulation_enrichment":
            start = time.time()
            identifier='identifier'
            groups=['group1', 'group2']
            annotation_col='annotation'
            reject_col='rejected'
            method='fisher'
            annotation_type = 'functional'
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

            if 'regulation_data' in self.args and 'annotation' in self.args:
                if self.args['regulation_data'] in self.data and self.args['annotation'] in self.data:
                    self.result[annotation_type+"_"+self.analysis_type] = analyses.run_regulation_enrichment(self.data[self.args['regulation_data']], self.data[self.args['annotation']], identifier=identifier, groups=groups, annotation_col=annotation_col, reject_col=reject_col, method=method)
            print('Enrichment', time.time() - start)
        elif self.analysis_type == 'long_format':
            self.result[self.analysis_type] = analyses.transform_into_long_format(self.data, drop_columns=self.args['drop_columns'], group=self.args['group'], columns=self.args['columns'])
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
                    self.result[self.analysis_type] = analyses.get_ranking_with_markers(self.data[self.args['data']], drop_columns=self.args['drop_columns'], group=self.args['group'], columns=self.args['columns'], list_markers=list_markers, annotation = annotations)
            print('Ranking', time.time() - start)
        elif self.analysis_type == 'coefficient_of_variation':
            self.result[self.analysis_type] = analyses.get_coefficient_variation(self.data, drop_columns=self.args['drop_columns'], group=self.args['group'], columns=self.args['columns'])
        elif self.analysis_type == 'publications_abstracts':
            self.result[self.analysis_type] = analyses.get_publications_abstracts(self.data, publication_col="publication", join_by=['publication','Proteins','Diseases'], index="PMID")
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
            self.result[self.analysis_type] = analyses.run_WGCNA(self.data, drop_cols_exp, drop_cols_cli, RsquaredCut=RsquaredCut, networkType=networkType, 
                                                            minModuleSize=minModuleSize, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro, merge_modules=merge_modules,
                                                            MEDissThres=MEDissThres, verbose=verbose)
            print('WGCNA', time.time() - start)
        elif self.analysis_type == 'multi_correlation':
            start = time.time()
            alpha = 0.05
            method = 'pearson'
            correction = ('fdr', 'indep')
            subject='subject'
            group='group'
            on=['subject', 'group']
            if 'on_cols' in self.args:
                on = self.args['on']
            if 'group' in self.args:
                group = self.args['group']
            if 'subject' in self.args:
                subject= self.args['subject']
            if "alpha" in self.args:
                alpha = self.args["args"]
            if "method" in self.args:
                method = self.args["method"]
            if "correction" in self.args:
                correction = self.args["correction"]
            self.result[self.analysis_type] = analyses.run_multi_correlation(self.data, alpha=alpha, subject=subject, group=group, on=on, method=method, correction=correction)
            print('multi-correlation', time.time() - start)
            

    def get_plot(self, name, identifier):
        plot = []
        if len(self.result) >=1:
            if name == "basicTable":
                colors = ('#C2D4FF','#F5F8FF')
                attr =  {'width':800, 'height':1500, 'font':12}
                subset = None
                figure_title = 'Basic table'
                if "colors" in self.args:
                    colors = self.args["colors"]
                if "attr" in self.args:
                    attr = self.args["attr"]
                if "subset" in self.args:
                    subset = self.args["subset"]
                if "title" in self.args:
                    figure_title = self.args["title"]
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args["title"] + id[0]+" vs "+id[1]
                    plot.append(figure.get_table(self.result[id], identifier, figure_title, colors=colors, subset=subset, plot_attr=attr))
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
                    plot.append(figure.get_barplot(self.result[id], identifier, self.args))
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
                    plot.append(figure.get_facet_grid_plot(self.result[id], identifier, self.args))
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
                    plot.append(figure.get_scatterplot(self.result[id], identifier, self.args))
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
                    plot.append(figure.get_pca_plot(self.result[id], identifier, self.args))
            elif name == "volcanoplot":
                alpha = 0.05
                lfc = 1.0
                if "alpha" in self.args:
                    alpha = self.args["alpha"]
                if "lfc" in self.args:
                    lfc = self.args["lfc"]
                for pair in self.result:
                    signature = self.result[pair]
                    self.args["title"] = self.args['title']+" "+pair[0]+" vs "+pair[1]
                    p = figure.run_volcano(signature, identifier+"_"+pair[0]+"_vs_"+pair[1], self.args)
                    plot.extend(p)
            elif name == 'network':
                source = 'source'
                target = 'target'
                if "source" in self.args:
                    source = self.args["source"]
                if "target" in self.args:
                    target = self.args["target"]
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args["title"] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args["title"]
                    self.args["title"] = figure_title
                    plot.append(figure.get_network(self.result[id], identifier, self.args))
            elif name == "heatmap":
                for id in self.result:
                    if not self.result[id].empty:
                        if isinstance(id, tuple):
                            identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                            figure_title = self.args["title"] + id[0]+" vs "+id[1]
                        else:
                            figure_title = self.args["title"]
                        self.args["title"] = figure_title
                        plot.append(figure.get_complex_heatmapplot(self.result[id], identifier, self.args))
            elif name == "mapper":
                for id in self.result:
                    labels = {}
                    if "labels" in self.args:
                        labels = self.args["labels"]
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    plot.append(figure.getMapperFigure(self.result[id], identifier, title=figure_title, labels=self.args["labels"]))
            elif name == "scatterplot_matrix":
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args["title"] = figure_title
                    plot.append(figure.get_scatterplot_matrix(self.result[id], identifier, self.args))
            elif name == "distplot":
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args["title"] = figure_title
                    plot.extend(figure.get_distplot(self.result[id], identifier, self.args))
            elif name == "violinplot":
                for id in self.result:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = self.args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = self.args['title']
                    self.args["title"] = figure_title
                    plot.extend(figure.get_violinplot(self.result[id], identifier, self.args))
            elif name == "wgcnaplots":
                start = time.time()
                data = {}
                input_data = self.data
                wgcna_data = self.result
                dfs = wgcna.get_data(input_data, drop_cols_exp=self.args['drop_cols_exp'], drop_cols_cli=self.args['drop_cols_cli'])
                if 'wgcna' in wgcna_data and wgcna_data['wgcna'] is not None and dfs is not None:
                    for dtype in wgcna_data['wgcna']:
                        data = {**dfs, **wgcna_data['wgcna'][dtype]}
                        plot.extend(figure.get_WGCNAPlots(data, identifier+"-"+dtype))
                print('WGCNA-plot', time.time() - start)
            elif name == 'ranking':
                for id in self.result:
                    plot.append(figure.get_ranking_plot(self.result[id], identifier, self.args))
            elif name == 'clustergrammer':
                for id in self.result:
                    plot.append(figure.get_clustergrammer_plot(self.result[id], identifier, self.args))
            elif name == 'cytonet':
                for id in self.result:
                    plot.append(figure.get_cytoscape_network(self.result[id], identifier, self.args))
            elif name == 'wordcloud':
                for id in self.result:
                    plot.append(figure.get_wordcloud(self.result[id], identifier, self.args))

        return plot
