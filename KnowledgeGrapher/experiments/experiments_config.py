
dataDirectory = "../../../data"
mappingFile = dataDirectory + "/ontologies/mapping.tsv"
experimentsImportDirectory = dataDirectory+"/imports/experiments"

#Dataset types
experimentsDir = dataDirectory + "/experiments/"
#Proteomics
clinicalDirectory = experimentsDir + "PROJECTID/clinical/"
proteomicsDirectory = experimentsDir + "PROJECTID/proteomics/"
genomicsDirectory = experimentsDir + "PROJECTID/wes/"
PTMdict = {"columns": ["Proteins",
                        "Positions within proteins", 
                        "Amino acid",
                        "Sequence window",
                        "Score", 
                        "Intensity \d+_AS\d+_?-?\d*", 
                        "Reverse",
                        "Potential contaminant",
                        "Score",
                        "Delta score",
                        "Score for localization",
                        "Localization prob"],
            "filters" : ["Reverse",
                        "Potential contaminant"],
            "attributes":{"col":["Sequence window",
                                "Score",
                                "Delta score",
                                "Score for localization",
                                "Localization prob"]
                                },
            "proteinCol" : "Proteins",
            "indexCol" : "Proteins",
            "valueCol" : "Intensity",
            "multipositions": "Positions within proteins",
            "positionCols": ["Positions within proteins","Amino acid"],
            "sequenceCol": "Sequence window",
            "geneCol":"Gene names",
            "log": "log10"}
ox = {"modId":"MOD:00256", "mod_acronym":"ox", "file":"Oxidation (M)Sites.txt"}
ox.update(PTMdict)
gly = {"modId":"MOD:00767", "mod_acronym":"gly", "file":"GlycationSites.txt"}
gly.update(PTMdict)
p = {"modId":"MOD:00696", "mod_acronym":"p", "file":"Phosphosites (STY).txt"}
p.update(PTMdict)
dataTypes = {"clinical":{"directory":clinicalDirectory,
                            "file":"clinicalData.xlsx"},
            "proteomics":{"directory": proteomicsDirectory,
                            "proteins":{"columns":
                                            ["Majority protein IDs",
                                            "Gene names", 
                                            "Q-value", 
                                            "Score", 
                                            "id",
                                            "LFQ intensity \d+_AS\d+_?-?\d*",  #subject_replicate_timepoint
                                            "Intensity \d+_AS\d+_?-?\d*",
                                            "Reverse",
                                            "Potential contaminant",
                                            "Only identified by site"],
                                        "filters" : ["Reverse",
                                                        "Potential contaminant",
                                                        "Only identified by site"],
                                        "proteinCol" : "Majority protein IDs",
                                        "valueCol" : "LFQ intensity",
                                        "indexCol" : "Majority protein IDs",
                                        "attributes":{  "cols":[
                                                                "Q-value",
                                                                "Score", 
                                                                "id"
                                                            ],
                                                        "regex":["Intensity"]
                                                    },
                                        "log": "log10",
                                        "file": "proteinGroups.txt"},
                            "peptides":{"columns":
                                            ["Sequence",
                                            "Amino acid before",
                                            "First amino acid",
                                            "Second amino acid",
                                            "Second last amino acid",
                                            "Last amino acid",
                                            "Amino acid after",
                                            "Experiment \d+_AS\d+_?-?\d*",
                                            "Proteins", 
                                            "Start position",
                                            "End position",
                                            "Score",
                                            "Protein group IDs",
                                            "Intensity \d+_AS\d+_?-?\d*", 
                                            "Reverse",
                                            "Potential contaminant"],
                                        "filters" : ["Reverse",
                                                        "Potential contaminant"],
                                        "attributes":{"col":["Score",
                                                            "Protein group IDs"]
                                                    },
                                        "proteinCol" : "Proteins",
                                        "valueCol" : "Intensity",
                                        "indexCol" : "Sequence",
                                        "positionCols":["Start position","End position"],
                                        "type": "tryptic peptide",
                                        "log": "log10",
                                        "file":"peptides.txt"},
                            "Oxydation(M)":ox,
                            "Glycation":gly,
                                "Phosphorylation":p
                            },
            "wes":{"directory": genomicsDirectory,
                        "columns" : ["Chr", "Start", "Ref", "Alt", 
                                    "Func.refGene", "Gene.refGene", 
                                    "ExonicFunc.refGene", "AAChange.refGene", 
                                    "Xref.refGene", "SIFT_score", "SIFT_pred", "Polyphen2_HDIV_score",
                                    "Polyphen2_HDIV_pred", "Polyphen2_HVAR_score",
                                    "Polyphen2_HVAR_pred", "LRT_score", "LRT_pred",
                                    "MutationTaster_score", "MutationTaster_pred",
                                    "MutationAssessor_score", "MutationAssessor_pred",
                                    "FATHMM_score", "FATHMM_pred", "PROVEAN_score",
                                    "PROVEAN_pred", "VEST3_score", "CADD_raw", "CLINSIG",
                                    "CLNDBN", "CLNACC", "CLNDSDB", "CLNDSDBID", "cosmic70", 
                                    "ICGC_Id", "ICGC_Occurrence"],
                        "position" : "Start",
                        "id_fields" : ["Chr", "Start", "Ref", "Alt"],
                        "alt_names" : "AAChange.refGene",
                        "somatic_mutation_attributes":["chr", "position", "reference", "alternative",
                                    "region", "gene",
                                    "function", "Xref", 
                                    "SIFT_score", "SIFT_pred", "Polyphen2_HDIV_score",
                                    "Polyphen2_HDIV_pred", "Polyphen2_HVAR_score",
                                    "Polyphen2_HVAR_pred", "LRT_score", "LRT_pred",
                                    "MutationTaster_score", "MutationTaster_pred",
                                    "MutationAssessor_score", "MutationAssessor_pred",
                                    "FATHMM_score", "FATHMM_pred", "PROVEAN_score",
                                    "PROVEAN_pred", "VEST3_score", "CADD_raw", "CLINSIG",
                                    "CLNDBN", "CLNACC", "CLNDSDB", "CLNDSDBID", "cosmic70", 
                                    "ICGC_Id", "ICGC_Occurrence", "alternative_names","ID"],
                        "new_columns": ["chr", "position", "reference", "alternative",
                                    "region", "gene",
                                    "function", "Xref", 
                                    "SIFT_score", "SIFT_pred", "Polyphen2_HDIV_score",
                                    "Polyphen2_HDIV_pred", "Polyphen2_HVAR_score",
                                    "Polyphen2_HVAR_pred", "LRT_score", "LRT_pred",
                                    "MutationTaster_score", "MutationTaster_pred",
                                    "MutationAssessor_score", "MutationAssessor_pred",
                                    "FATHMM_score", "FATHMM_pred", "PROVEAN_score",
                                    "PROVEAN_pred", "VEST3_score", "CADD_raw", "CLINSIG",
                                    "CLNDBN", "CLNACC", "CLNDSDB", "CLNDSDBID", "cosmic70", 
                                    "ICGC_Id", "ICGC_Occurrence", "sample", "variantCallingMethod", "annotated", "alternative_names", "ID"],
                        "types" :{"Somatic_mutation": "KEEP", "Germline_mutation": "REJECT"}                       
                }
        }
