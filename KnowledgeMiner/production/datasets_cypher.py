queries = {"proteomics" : {
            "NUMBER_OF_PROTEINS":
                '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEIN]->(protein:Protein) 
                    WHERE project.id="PROJECTID" 
                    RETURN a.id AS name, a.id AS x,COUNT(DISTINCT(protein.id)) AS y, a.group AS group ORDER BY group;''',
            "NUMBER_OF_PEPTIDES":
                '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PEPTIDE]->(peptide:Peptide) 
                    WHERE project.id="PROJECTID" 
                    RETURN a.id AS name, a.id AS x, a.group AS group,COUNT(DISTINCT(peptide.id)) AS y ORDER BY group;''',
            "NUMBER_OF_MODIFIED_PROTEINS":
                '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEINMODIFICATION]->(modifiedprotein:Modified_protein) 
                    WHERE project.id="PROJECTID" 
                    RETURN a.id AS name, a.id AS x,a.group AS group,COUNT(DISTINCT(modifiedprotein.id)) AS y ORDER BY group;''',
            "DATASET": 
                '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEIN]->(protein:Protein) 
                    WHERE project.id="PROJECTID" 
                    RETURN a.id AS sample, protein.id AS identifier, a.group AS group, toFloat(r.value) as LFQ_intensity, protein.name AS name ORDER BY group;'''
                }
            }
