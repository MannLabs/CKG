stats_file = "../../data/imports/stats/stats.hdf"

logo = '../static/img/logo.png'

footer = '<div id="PageFooter"><p> The Clinical Knowledge Graph has been implemented by <a href="mailto:alberto.santos@cpr.ku.dk">Alberto Santos</a>, <a>Annelaura B. Nielsen</a> and <a href="mailto:rita.colaco@cpr.ku.dk">Ana R. Colaço</a></p></br> <p>This tool is used by the Clinical Proteomics Department, Prof. Matthias Mann, at <a href="http://www.cpr.ku.dk">Novo Nordisk Foundation Center for Protein Reseach</a></p></div>'

projectPage = {"overview":[
                            ("overview", [], ["basicTable"], {}), 
                            ("number_subjects", [], ["basicTable"], {}), 
                            ("number_analytical_samples", [], ["basicTable"], {})
                            ]
                }
proteomicsPage= {"overview":[
                            ("number_peptides_analytical_sample", [], ["basicBarPlot"], {"x_title":"Analytical sample", "y_title":"number of peptides"}),
                            ("number_proteins_analytical_sample", [], ["basicBarPlot"], {"x_title":"Analytical sample", "y_title":"number of proteins"}),
                            ("number_modified_proteins_analytical_sample", [], ["basicBarPlot"], {"x_title":"Analytical sample", "y_title":"number of modified proteins"})
                            ],
                "stratification":[
                                ("identified_proteins_sample_group", 
                               ["pca", "tsne", "umap"], 
                                ["scatterPlot"], 
                                {"imputation":True, "imputation_method":"Mixed", "x_title":"PC1", "y_title":"PC2", "components":2, "perplexity":40, "n_iter":1000, "init":'pca'})
                                ],
                 "regulation":[
                                ("identified_proteins_sample_group_with_gene",
                                ["ttest"], 
                                ["volcanoPlot", "basicTable"],
                                {"imputation":True, "imputation_method":"Mixed", "alpha":0.05, "drop_cols":["sample","gene_name"], "name":"name"})
                             ],
                #"correlation":[
                 #               ("identified_proteins_sample_group", 
                 #               ["correlation"], 
                 #               ["3Dnetwork", "basicTable"],
                 #               {"source":"node1", "target":"node2"})
                 #              ],
                "action":[
                           ("drug_acts_on_proteins", 
                           [],
                           ["basicTable", "3Dnetwork"],
                           {"replace":[("ACTION","inhibition"),("PROTEINS",'"O60341"')]})
                          ]
                    }
wesPage= {"overview":[
                        ("number_somatic_mutations_by_type_analytical_sample",["basicBarPlot", "basicTable"]),
                    ],
        "targets":[
                        ("target_analysis_variants",["basicTable", "3dNetwork"])
                    ]
        }


### Project Page configuration
pages = {"projectPage":{
                "project": projectPage,
                "proteomics": proteomicsPage,
                #"wes": wesPage
                },
            "importsPage":{"stats_file":stats_file},
            "projectCreationPage":{}
            }
## Overview 
## Project Name
## Project description
## Studied disease
## Person responsible
## Participants
## Number of enrolled subjects




