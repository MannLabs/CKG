###### PhosphoSitePlus Database ########
#Registration necessary to download files, automatic download not possible

modifications = {"ac":"MOD:00394",
                 "m1":"MOD:00599",
                 "m2":"MOD:00429",
                 "m3":"MOD:00430",
                 "me":"MOD:00427",
                 "ga":"MOD:00563",
                 "gl":"MOD:00448",
                 "sm":"MOD:01149",
                 "ub":"MOD:01148",
                 "p":"MOD:01456",
                 "ox":"MOD:00256",
                 "gly":"MOD:00767"}

annotation_files = {("disease", "associated_with"):"Disease-associated_sites.gz",
         ("modified_protein", "is_substrate_of"):"Kinase_Substrate_Dataset.gz",
         ("biological_process", "associated_with"):"Regulatory_sites.gz"}


headers = {"disease":['START_ID', 'END_ID', 'TYPE', 'evidence_type','score','source', 'publications'],
         "modified_protein":['START_ID', 'END_ID', 'TYPE', 'evidence_type','score','source'],
         "biological_process":['START_ID', 'END_ID', 'TYPE', 'evidence_type','score','source', 'publications', 'action']}

