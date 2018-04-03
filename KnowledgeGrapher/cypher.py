CREATE_DB = "LOAD"

CREATE_USER = "CALL dbms.security.createUser(username, password, requirePasswordChange = False)"
ADD_ROLE_TO_USER = "CALL dbms.security.addRoleToUser(rolename, username)"

COUNT_RELATIONSHIPS = "MATCH (:ENTITY1)-[:RELATIONSHIP]->(:ENTITY2) return count(*) AS count;"
REMOVE_RELATIONSHIPS = "MATCH (:ENTITY1)-[r:RELATIONSHIP]->(:ENTITY2) delete r;"

IMPORT_ONTOLOGY_DATA = '''CREATE INDEX ON :ENTITY(name);
                        CREATE CONSTRAINT ON (e:ENTITY) ASSERT e.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY.csv" AS line
                        MERGE (e:ENTITY {id:line.ID})
                        ON CREATE SET e.name=line.name,e.description=line.description,e.type=line.type,e.synonyms=SPLIT(line.synonyms,','); 
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY_has_parent.csv" AS line
                        MATCH (e1:ENTITY {id:line.START_ID})
                        MATCH (e2:ENTITY {id:line.END_ID}) 
                        CREATE UNIQUE (e1)-[:HAS_PARENT]->(e2);
                        '''

IMPORT_PROTEIN_DATA = '''CREATE INDEX ON :Protein(accession); 
                        CREATE CONSTRAINT ON (a:Protein) ASSERT a.id IS UNIQUE; 
                        CREATE CONSTRAINT ON (a:Protein) ASSERT a.accesion IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/Protein.csv" AS line
                        MERGE (p:Protein {id:line.ID}) 
                        ON CREATE SET p.accession=line.accession,p.name=line.name,p.description=line.description,p.taxid=line.taxid,p.synonyms=SPLIT(line.synonyms,',');
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/uniprot_gene_translated_into.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (g:Gene {id:line.END_ID})
                        CREATE UNIQUE (g)-[:TRANSLATED_INTO]->(p);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/uniprot_transcript_translated_into.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (t:Transcript {id:line.END_ID})
                        CREATE UNIQUE (t)-[:TRANSLATED_INTO]->(p);
                        '''

IMPORT_GENE_DATA = '''CREATE CONSTRAINT ON (g:Gene) ASSERT g.id IS UNIQUE; 
                        CREATE CONSTRAINT ON (g:Gene) ASSERT g.name IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/Gene.csv" AS line
                        MERGE (g:Gene {id:line.ID}) 
                        ON CREATE SET g.name=line.name,g.family=line.family,g.taxid=line.taxid,g.synonyms=SPLIT(line.synonyms,','); 
                        '''

IMPORT_TRANSCRIPT_DATA = '''CREATE CONSTRAINT ON (t:Transcript) ASSERT t.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/Transcript.csv" AS line
                        MERGE (t:Transcript {id:line.ID}) 
                        ON CREATE SET t.name=line.name,t.class=line.class,t.taxid=line.taxid,t.assembly=line.assembly; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/refseq_located_in.csv" AS line
                        MATCH (t:Transcript {id:line.START_ID})
                        MATCH (c:Chromosome {id:line.END_ID})
                        CREATE UNIQUE (t)-[:LOCATED_IN{start:line.start,end:line.end,strand:line.strand}]->(c);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/refseq_transcribed_into.csv" AS line
                        MATCH (g:Gene {id:line.START_ID})
                        MATCH (t:Transcript {id:line.END_ID})
                        CREATE UNIQUE (g)-[:TRANSCRIBED_INTO]->(t);
                        '''

IMPORT_CHROMOSOME_DATA = '''CREATE CONSTRAINT ON (c:Chromosome) ASSERT c.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/Chromosome.csv" AS line
                        MERGE (c:Chromosome {id:line.ID})
                        ON CREATE SET c.name=line.name,c.taxid=line.taxid;
                        '''

IMPORT_CURATED_PPI_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/RESOURCE_interacts_with.csv" AS line
                        MATCH (p1:Protein {id:line.START_ID})
                        MATCH (p2:Protein {id:line.END_ID})
                        CREATE UNIQUE (p1)-[:CURATED_INTERACTS_WITH{score:line.score,interaction_type:line.interaction_type,method:SPLIT(line.method,','),source:SPLIT(line.source,','),publications:SPLIT(line.publications,',')}]->(p2);'''
IMPORT_COMPILED_PPI_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/RESOURCE_interacts_with.csv" AS line
                        MATCH (p1:Protein {id:line.START_ID})
                        MATCH (p2:Protein {id:line.END_ID})
                        CREATE UNIQUE (p1)-[:COMPILED_INTERACTS_WITH{score:line.score,interaction_type:line.interaction_type,method:SPLIT(line.method,','),source:SPLIT(line.source,','),scores:SPLIT(line.evidences,','),evidences:SPLIT(line.evidences,',')}]->(p2);'''

IMPORT_DISEASE_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/RESOURCE_associated_with.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (d:Disease {id:line.END_ID})
                        CREATE UNIQUE (p)-[:ASSOCIATED_WITH{score:line.score,evidence_type:line.evidence_type,source:line.source,number_publications:line.number_publications}]->(d);'''

IMPORT_CURATED_DRUG_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/RESOURCE_targets.csv" AS line
                        MATCH (d:Drug {id:line.START_ID})
                        MATCH (g:Gene {id:line.END_ID})
                        CREATE UNIQUE (d)-[:CURATED_TARGETS{source:line.source,interaction_type:line.interaction_type, score: line.score}]->(g);'''

IMPORT_COMPILED_DRUG_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/RESOURCE_targets.csv" AS line
                        MATCH (d:Drug {id:line.START_ID})
                        MATCH (g:Gene {id:line.END_ID})
                        CREATE UNIQUE (d)-[:COMPILED_TARGETS{score:line.score, source:line.source,interaction_type:line.interaction_type,scores:SPLIT(line.evidences,','),evidences:SPLIT(line.evidences,',')}]->(g);'''

IMPORT_DATASETS = {"proteomics":'''USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_proteins.csv" AS line
                        MATCH (s:Analytical_sample {id:line.START_ID}) 
                        MATCH (p:Protein{id:line.END_ID}) 
                        CREATE UNIQUE (s)-[:HAS_QUANTIFIED_PROTEIN{value:line.value}]->(p);
                        CREATE CONSTRAINT ON (p:Peptide) ASSERT p.id IS UNIQUE;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_peptides.csv" AS line
                        MERGE (p:Peptide{id:line.ID}) 
                        ON CREATE SET p.type=line.type;
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_peptide_protein.csv" AS line 
                        MATCH (p1:Peptide {id:line.START_ID})
                        MATCH (p2:Protein {id:line.END_ID}) 
                        CREATE UNIQUE (p1)-[:BELONGS_TO_PROTEIN]->(p2);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_subject_peptide.csv" AS line 
                        MATCH (s:Analytical_sample {id:line.START_ID})
                        MATCH (p:Peptide {id:line.END_ID}) 
                        CREATE UNIQUE (s)-[:HAS_QUANTIFIED_PEPTIDE{value:line.value}]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_protein_modification.csv" AS line 
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (m:PTM {id:line.END_ID}) 
                        CREATE UNIQUE (p)-[:HAS_MODIFICATION{position:line.position,residue:line.residue}]->(m);
                        CREATE CONSTRAINT ON (m:Modified_protein) ASSERT m.id IS UNIQUE;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_modifiedprotein.csv" AS line
                        MERGE (m:Modified_protein {id:line.ID}) 
                        ON CREATE SET m.protein=line.protein,m.position=line.position,m.residue=line.residue;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_modifiedprotein_modification.csv" AS line
                        MATCH (mp:Modified_protein {id:line.START_ID})
                        MATCH (p:PTM {id:line.END_ID}) 
                        CREATE UNIQUE (mp)-[:MODIFIED_WITH]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_modifiedprotein_protein.csv" AS line 
                        MATCH (mp:Modified_protein {id:line.START_ID})
                        MATCH (p:Protein {id:line.END_ID}) 
                        CREATE UNIQUE (mp)-[:BELONGS_TO_PROTEIN]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_modifiedprotein_subject.csv" AS line
                        MATCH (s:Analytical_sample {id:line.START_ID})
                        MATCH (mp:Modified_protein {id:line.END_ID}) 
                        CREATE UNIQUE (s)-[:HAS_QUANTIFIED_PROTEINMODIFICATION{value:line.value}]->(mp);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_clinical.csv" AS line 
                        MATCH (s:Analytical_sample {id:line.START_ID})
                        MATCH (c:Clinical_measurement {id:line.END_ID}) 
                        CREATE UNIQUE (s)-[:HAS_QUANTIFIED_CLINICAL{value: line.value}]->(c);'''
                }

CREATE_PROJECT = '''CREATE CONSTRAINT ON (p:Project) ASSERT p.id IS UNIQUE;
                    CREATE CONSTRAINT ON (p:Project) ASSERT p.name IS UNIQUE; 
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID.csv" AS line
                    MERGE (p:Project {id:line.ID}) 
                    ON CREATE SET p.name=line.name,p.description=line.description,p.acronym=line.acronym,p.responsible=line.responsible;
                    '''
CREATE_SUBJECTS = '''CREATE CONSTRAINT ON (s:Subject) ASSERT s.id IS UNIQUE;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_subjects.csv" AS line
                    MERGE (s:Subject {id:line.ID}) 
                    ON CREATE SET s.external_id=line.external_id;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_project.csv" AS line 
                    MATCH (p:Project {id:line.START_ID})
                    MATCH (s:Subject {id:line.END_ID}) 
                    CREATE UNIQUE (p)-[:HAS_ENROLLED]->(s);
                    '''

CREATE_BIOSAMPLES = '''CREATE CONSTRAINT ON (s:Biological_sample) ASSERT s.id IS UNIQUE; 
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_biological_samples.csv" AS line
                    MERGE (s:Biological_sample {id:line.ID})
                    ON CREATE SET s.source=line.source,s.external_id=line.external_id,s.owner=line.owner,s.collection_date=line.collection_date,s.conservation_conditions=line.conservation_conditions,s.storage=line.storage,s.status=line.status,s.quantity=line.quantity,s.quantity_units=line.quantity_units;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_subject_biosample.csv" AS line 
                    MATCH (b:Biological_sample {id:line.START_ID}) 
                    MATCH (s:Subject {id:line.END_ID})
                    CREATE UNIQUE (b)-[:BELONGS_TO_SUBJECT]->(s);'''

CREATE_ANALYTICALSAMPLES = '''CREATE INDEX ON :Analytical_sample(group);
                    CREATE CONSTRAINT ON (s:Analytical_sample) ASSERT s.id IS UNIQUE; 
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_analytical_samples.csv" AS line
                    MERGE (s:Analytical_sample {id:line.ID})
                    ON CREATE SET s.collection_date=line.collection_date,s.conservation_conditions=line.conservation_conditions,s.storage=line.storage,s.status=line.status,s.quantity=line.quantity,s.quantity_units=line.quantity_units,s.group=line.group;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_biosample_analytical.csv" AS line
                    MATCH (s1:Biological_sample {id:line.START_ID})
                    MATCH (s2:Analytical_sample {id:line.END_ID}) 
                    CREATE UNIQUE (s1)-[:SPLITTED_INTO{quantity:line.quantity,quantity_units:line.quantity_units}]->(s2);'''

IMPORT_PROJECT = [CREATE_PROJECT, CREATE_SUBJECTS, CREATE_BIOSAMPLES, CREATE_ANALYTICALSAMPLES]
