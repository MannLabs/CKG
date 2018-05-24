

logo = '../static/img/logo.png'

footer = '<div id="PageFooter"><p> The Clinical Knowledge Graph has been implemented by <a href="mailto:alberto.santos@cpr.ku.dk">Alberto Santos</a>, <a>Annelaura B. Nielsen</a> and <a href="mailto:rita.colaco@cpr.ku.dk">Ana R. Colaço</a></p></br> <p>This tool is used by the Clinical Proteomics Department, Prof. Matthias Mann, at <a href="http://www.cpr.ku.dk">Novo Nordisk Foundation Center for Protein Reseach</a></p></div>'


proteomicsPage= {"overview":[
                                ("numberProteinsAnalyticalSample",["basicBarPlot"]),
                                ("numberPepetidesAnlyticalSample", ["basicBarPlot"]),
                                ("numberModifiedProteinsAnalyticalSample", ["basicPlot"])
                                ],
                    "stratification":[
                                ("pca", "pcaPlot")
                                ],
                    "correlation":[
                                ("correlationAnalysis", ["complexHeatmapPlot", "3dNetwork"])
                                ],
                    "regulation":[
                                ("differentialRegulation", ["volcanoPlot", "basicTable", "ppiNetwork"])
                                ],
                    "targets":[
                                ("targetAnalysisGenes", ["basicTable", "3dNetwork"])
                                ]
                    }
wesPage= {"overview":[
                        ("numberSomaticMutationsByTypeAnalyticalSample",["basicBarPlot", "basicTable"]),
                    ],
        "targets":[
                        ("targetAnalysisVariants",["basicTable", "3dNetwork"])
                    ]
        }


### Project Page configuration
projectPage= {"overview":[
                            ("projectOverview", )
                        ],
                "proteomics": proteomicsPage,
                "wes": wesPage
            }
## Overview 
## Project Name
## Project description
## Studied disease
## Person responsible
## Participants
## Number of enrolled subjects




