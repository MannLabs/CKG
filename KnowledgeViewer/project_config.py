configuration = {"proteomics":
                {"overview":[ #section
                            ("number of peptides", #section_query
                            [], #analysis_types
                            ["basicBarPlot"], #plot_type
                            {"x_title":"Analytical sample", "y_title":"number of peptides"}), #args
                            ("number of proteins",
                            [],
                            ["basicBarPlot"],
                            {"x_title":"Analytical sample", "y_title":"number of proteins"}),
                            ("number of modified proteins",
                            [],
                            ["basicBarPlot"],
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
                                ["basicTable"],
                                {"imputation":True, "imputation_method":"mixed", "alpha":0.05, "drop_cols":["sample","gene_name"], "name":"name"})
                             ],
                "correlation":[
                                ("regulation", 
                                ["correlation"], 
                                ["3Dnetwork", "basicTable"],
                                {"source":"node1", "target":"node2"})
                               ],
                
                    }
            }
