queries = {
    "NUMBER_PROTEINS_ANALYTICAL_SAMPLE":("Number of Proteins", 
            '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEIN]->(protein:Protein) WHERE project.id="PROJECTID" RETURN a.id,a.group,protein.id,r.value;'''),
    "NUMBER_PEPTIDES_ANALYTICAL_SAMPLE":("Number of Peptides", 
            '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PEPTIDE]->(peptide:Peptide) WHERE project.id="PROJECTID" RETURN a.id,a.group,peptide.id,r.value;'''),
    "NUMBER_MODIFIED_PROTEINS_ANALYTICAL_SAMPLE":("Number of Modified Proteins", 
            '''MATCH p=(project:Project)-[*3]-(a:Analytical_sample)-[r:HAS_QUANTIFIED_PROTEINMODIFICATION]->(modifiedprotein:Modified_protein) WHERE project.id="P0000001" RETURN a.id,a.group,modifiedprotein.id,r.value;''')
}
