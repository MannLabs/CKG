"proteomics":
    "NUMBER_OF_PROTEINS":
        'name': 'number of proteins'
        'description': 'Extracts the number of proteins identified in a given Project. Requires: Project.id'
        'involves_nodes':
            - 'Project'
            - 'Analytical_sample'
            - 'Protein'
        'involves_rels':
            - 'HAS_QUANTIFIED_PROTEIN'
        'query': >
            MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEIN]->(protein:Protein) 
            WHERE project.id="PROJECTID" 
            RETURN a.id AS name, a.id AS x,COUNT(DISTINCT(protein.id)) AS y, a.group AS group ORDER BY group;
    "NUMBER_OF_PEPTIDES":
        'name': 'number of peptides'
        'description': 'Extracts the number of peptides identified in a given Project. Requires: Project.id'
        'involves_nodes':
            - 'Project'
            - 'Analytical_sample'
            - 'Peptide'
        'involves_rels':
            - 'HAS_QUANTIFIED_PEPTIDE'
        'query': >
            MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PEPTIDE]->(peptide:Peptide) 
            WHERE project.id="PROJECTID" 
            RETURN a.id AS name, a.id AS x, a.group AS group,COUNT(DISTINCT(peptide.id)) AS y ORDER BY group;
    "NUMBER_OF_MODIFIED_PROTEINS":
        'name': 'number of modified proteins'
        'description': 'Extracts the number of modified proteins identified in a given Project. Requires: Project.id'
        'involves_nodes':
            - 'Project'
            - 'Analytical_sample'
            - 'Modified_protein'
        'involves_rels':
            - 'HAS_QUANTIFIED_MODIFIED_PROTEIN'
        'query': >
            MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_MODIFIED_PROTEIN]->(modifiedprotein:Modified_protein) 
            WHERE project.id="PROJECTID" 
            RETURN a.id AS name, a.id AS x,a.group AS group,COUNT(DISTINCT(modifiedprotein.id)) AS y ORDER BY group;
    "DATASET":
        'name': 'get dataset from project'
        'description': 'Extracts the dataset matrix of quantified proteins in a given Project. Requires: Project.id'
        'involves_nodes':
            - 'Project'
            - 'Analytical_sample'
            - 'Protein'
        'involves_rels':
            - 'HAS_QUANTIFIED_PROTEIN'
        'query': >
            MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEIN]->(protein:Protein) 
            WHERE project.id="PROJECTID" 
            RETURN a.id AS sample, protein.id AS identifier, a.group AS group, toFloat(r.value) as LFQ_intensity, protein.name AS name ORDER BY group;
