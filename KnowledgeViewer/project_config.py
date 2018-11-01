configuration = {"proteomics":
                    {"overview":[ #section
                                ("number of peptides", #section_query
                                [], #analysis_types
                                ["basicBarPlot", "basicTable"], #plot_type
                                {"x_title":"Analytical sample", "y_title":"number of peptides"}), #args
                                ("number of proteins",
                                [],
                                ["basicBarPlot", "basicTable"],
                                {"x_title":"Analytical sample", "y_title":"number of proteins"}),
                                ("number of modified proteins",
                                [],
                                ["basicBarPlot", "basicTable"],
                                {"x_title":"Analytical sample", "y_title":"number of modified proteins"})
                                ],
                    "stratification":[
                                    ("preprocessed", 
                                   ["pca", "tsne", "umap"], 
                                    ["scatterPlot"], 
                                    {"imputation":True, "imputation_method":"mixed", "x_title":"PC1", "y_title":"PC2", "components":2, "perplexity":40, "n_iter":1000, "init":'pca'})
                                    ],
                     "regulation":[
                                    ("preprocessed",
                                    ["anova"], 
                                    ["basicTable", "volcanoPlot"],
                                    {"imputation":True, "imputation_method":"mixed", "alpha":0.05, "drop_cols":["sample","gene_name"], "name":"name"})
                                 ],
                    "correlation":[
                                    ("regulated", 
                                    ["correlation"], 
                                    ["heatmap", "3Dnetwork", "basicTable"],
                                    {"source":"node1", "target":"node2"})
                                   ],
                    "mapper":[
                                ("regulated",
                                ["mapper"],
                                ["mapper"],
                                {"n_cubes":15, "overlap":0.5, "n_clusters":3, "linkage":"complete", "affinity":"correlation"})
                            ],
                    
                        }
            }
