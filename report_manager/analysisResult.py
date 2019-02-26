import report_manager.analyses.basicAnalysis as analyses
from report_manager.plots import basicFigures as figure
import itertools


class AnalysisResult:
    def __init__(self, identifier, analysis_type, args, data, result=None):
        self._identifier = identifier
        self._analysis_type = analysis_type
        self._args = args
        self._data = data
        self._result = result
        if self._result is None:
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
    def identifier(self, analysis_type):
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
        result, args = self.get_analysis_result()
        self.result = result
        self.args = args

    def get_analysis_result(self):
        result = {}
        args = self.args
        if self.analysis_type == "long_format":
            r = analyses.transform_into_long_format(self.data, args['index'], args['x'],
                                                                args['y'], extra=[args['group']], use_index=args['use_index'])
            result[self.analysis_type] = r
        if self.analysis_type == "pca":
            components = 2
            if "components" in args:
                components = args["components"]
            result, nargs = analyses.run_pca(self.data, components=components)
            args.update(nargs)
        elif self.analysis_type  == "tsne":
            components = 2
            perplexity = 40
            n_iter = 1000
            init='pca'
            if "components" in args:
                components = args["components"]
            if "perplexity" in args:
                perplexity = args["perplexity"]
            if "n_iter" in args:
                n_iter = args["n_iter"]
            if "init" in args:
                init = args["init"]
            result, nargs = analyses.run_tsne(self.data, components=components, perplexity=perplexity, n_iter=n_iter, init=init)
            args.update(nargs)
        elif self.analysis_type  == "umap":
            n_neighbors=10
            min_dist=0.3
            metric='cosine'
            if "n_neighbors" in args:
                n_neighbors = args["n_neighbors"]
            if "min_dist" in args:
                min_dist = args["min_dist"]
            if "metric" in args:
                metric = args["metric"]
            if n_neighbors < self.data.shape[0]:
                result, nargs = analyses.run_umap(self.data, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
                args.update(nargs)
        elif self.analysis_type  == "mapper":
            n_cubes = 15
            overlap = 0.5
            n_clusters = 3
            linkage = "complete"
            affinity = "correlation"
            labels = {}
            if "labels" in args:
                labels = args["labels"]
            if "n_cubes" in args:
                n_cubes = args["n_cubes"]
            if "overlap" in args:
                overlap = args["overlap"]
            if "n_clusters" in args:
                n_clusters = args["n_clusters"]
            if "linkage" in args:
                linkage = args["linkage"]
            if "affinity" in args:
                affinity = args["affinity"]
            r, nargs = analyses.runMapper(self.data, n_cubes=n_cubes, overlap=overlap, n_clusters=n_clusters, linkage=linkage, affinity=affinity)
            args.update(nargs)
            result[self.analysis_type] = r
        elif self.analysis_type  == 'ttest':
            alpha = 0.05
            if "alpha" in args:
                alpha = args["alpha"]
            for pair in itertools.combinations(self.data.group.unique(),2):
                ttest_result = analyses.ttest(self.data, pair[0], pair[1], alpha = 0.05)
                result[pair] = ttest_result
        elif self.analysis_type  == 'anova':
            alpha = 0.05
            drop_cols = []
            group = 'group'
            permutations = 50
            if "alpha" in args:
                alpha = args["alpha"]
            if "drop_cols" in args:
                drop_cols = args['drop_cols']
            if "group" in args:
                group = args["group"]
            if "permutations" in args:
                permutations = args["permutations"]
            anova_result = analyses.anova(self.data, drop_cols=drop_cols, group=group, alpha=alpha, permutations=permutations)
            result[self.analysis_type] = anova_result
        elif self.analysis_type == "repeated_measurements_anova":
            alpha = 0.05
            drop_cols = []
            group = 'group'
            sample = 'sample'
            permutations = 50
            if "alpha" in args:
                alpha = args["alpha"]
            if "drop_cols" in args:
                drop_cols = args['drop_cols']
            if "group" in args:
                group = args["group"]
            if "sample" in args:
                sample = args["sample"]
            if "permutations" in args:
                permutations = args["permutations"]
            anova_result = analyses.repeated_measurements_anova(self.data, drop_cols=drop_cols, sample=sample, group=group, alpha=alpha, permutations=permutations)
            result[self.analysis_type] = anova_result
        elif self.analysis_type  == "correlation":
            alpha = 0.05
            method = 'pearson'
            correction = ('fdr', 'indep')
            if "alpha" in args:
                alpha = args["args"]
            if "method" in args:
                method = args["method"]
            if "correction" in args:
                correction = args["correction"]
            result[self.analysis_type] = analyses.runCorrelation(self.data, alpha=alpha, method=method, correction=correction)
        elif self.analysis_type == "interaction":
            result[self.analysis_type], nargs = analyses.get_interaction_network(self.data)
            args.update(nargs)
        elif self.analysis_type == "wgcna":
            # filename_exp = '/Users/plh450/Clinical_Proteomics/CKG/WGCNA/proteomics.h5'
            # key_exp = 'preprocessed'
            # filename_cli = '/Users/plh450/Clinical_Proteomics/CKG/WGCNA/clinical.h5'
            # key_cli = 'preprocessed'
            drop_cols_exp = []
            drop_cols_cli = []
            RsquaredCut = 0.8
            networkType = 'unsigned'
            network_verbose = 2
            minModuleSize = 30
            deepSplit = 2
            pamRespectsDendro = False
            merge_modules = True
            MEDissThres = 0.25
            verbose_merge = 3
            if "drop_cols_exp" in args:
                drop_cols_exp = args['drop_cols_exp']
            if "drop_cols_cli" in args:
                drop_cols_cli = args['drop_cols_cli']
            if "RsquaredCut" in args:
                RsquaredCut = args["RsquaredCut"]
            if "networkType" in args:
                networkType = args["networkType"]
            if "network_verbose" in args:
                network_verbose = args["network_verbose"]
            if "minModuleSize" in args:
                minModuleSize = args["minModuleSize"]
            if "deepSplit" in args:
                deepSplit = args["deepSplit"]
            if "pamRespectsDendro" in args:
                pamRespectsDendro = args["pamRespectsDendro"]
            if "merge_modules" in args:
                merge_modules = args["merge_modules"]
            if "MEDissThres" in args:
                MEDissThres = args["MEDissThres"]
            if "verbose_merge" in args:
                verbose_merge = args["verbose_merge"]
            result[self.analysis_type] = analyses.runWGCNA(self.data, drop_cols_exp, drop_cols_cli, RsquaredCut=0.8, networkType=networkType, network_verbose=network_verbose,
                                                            minModuleSize=minModuleSize, deepSplit=deepSplit, pamRespectsDendro=pamRespectsDendro, merge_modules=merge_modules,
                                                            MEDissThres=MEDissThres, verbose_merge=verbose_merge)
        return result, args

    def get_plot(self, name, identifier):
        data = self.result
        args = self.args
        plot = []
        if len(data) >=1:
            if name == "basicTable":
                colors = ('#C2D4FF','#F5F8FF')
                attr =  {'width':800, 'height':1500, 'font':12}
                subset = None
                if "colors" in args:
                    colors = args["colors"]
                if "attr" in args:
                    attr = args["attr"]
                if "subset" in args:
                    subset = args["subset"]
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args["title"] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args["title"]
                    plot.append(figure.getBasicTable(data[id], identifier, figure_title, colors=colors, subset=subset, plot_attr=attr))
            elif name == "barplot":
                x_title = "x"
                y_title = "y"
                if "x_title" in args:
                    x_title = args["x_title"]
                if "y_title" in args:
                    y_title = args["y_title"]
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args['title']
                    args["title"] = figure_title
                    plot.append(figure.get_barplot(data[id], identifier, args))
            elif name == "facetplot":
                x_title = "x"
                y_title = "y"
                plot_type = "bar"
                if "x_title" not in args:
                    args["x_title"] = x_title
                if "y_title" not in args:
                    args["y_title"] = y_title
                if "plot_type" not in args:
                    args["plot_type"] = plot_type
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args['title']
                    args['title'] = figure_title
                    plot.append(figure.get_facet_grid_plot(data[id], identifier, args))
            elif name == "scatterplot":
                x_title = "x"
                y_title = "y"
                if "x_title" in args:
                    x_title = args["x_title"]
                if "y_title" in args:
                    y_title = args["y_title"]
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args['title']
                    args['title'] = figure_title
                    plot.append(figure.get_scatterplot(data[id], identifier, args))
            elif name == "volcanoplot":
                alpha = 0.05
                lfc = 1.0
                if "alpha" in args:
                    alpha = args["alpha"]
                if "lfc" in args:
                    lfc = args["lfc"]
                for pair in data:
                    signature = data[pair]
                    args["title"] = args['title']+" "+pair[0]+" vs "+pair[1]
                    p = figure.run_volcano(signature, identifier+"_"+pair[0]+"_vs_"+pair[1], args)
                    plot.extend(p)
            elif name == 'network':
                source = 'source'
                target = 'target'
                if "source" in args:
                    source = args["source"]
                if "target" in args:
                    target = args["target"]
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args["title"] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args["title"]
                    args["title"] = figure_title
                    plot.append(figure.get_network(data[id], identifier, args))
            elif name == "heatmap":
                for id in data:
                    if not data[id].empty:
                        if isinstance(id, tuple):
                            identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                            figure_title = args["title"] + id[0]+" vs "+id[1]
                        else:
                            figure_title = args["title"]
                        args["title"] = figure_title
                        plot.append(figure.get_complex_heatmapplot(data[id], identifier, args))
            elif name == "mapper":
                for id in data:
                    labels = {}
                    if "labels" in args:
                        labels = args["labels"]
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args['title']
                    plot.append(figure.getMapperFigure(data[id], identifier, title=figure_title, labels=args["labels"]))
            elif name == "scatterplot_matrix":
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args['title']
                    args["title"] = figure_title
                    plot.append(figure.get_scatterplot_matrix(data[id], identifier, args))
            elif name == "distplot":
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args['title']
                    args["title"] = figure_title
                    plot.extend(figure.get_distplot(data[id], identifier, args))
            elif name == "violinplot":
                for id in data:
                    if isinstance(id, tuple):
                        identifier = identifier+"_"+id[0]+"_vs_"+id[1]
                        figure_title = args['title'] + id[0]+" vs "+id[1]
                    else:
                        figure_title = args['title']
                    args["title"] = figure_title
                    plot.extend(figure.get_violinplot(data[id], identifier, args))
            elif name == "wgcnaplots":
                for id in data:
                    plot.extend(figure.get_WGCNAPlots(data[id], identifier))

        return plot
