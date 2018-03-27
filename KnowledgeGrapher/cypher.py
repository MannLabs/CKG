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

IMPORT_DATASETS = {"proteomicsdata":'''USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_proteomicsdata.csv" AS line 
                        MATCH (s:Sample{id:line.START_ID})
                        MATCH (p:Protein {id:line.END_ID)}) 
                        CREATE UNIQUE (s)-[:HAS_QUANTIFIED_PROTEIN{quantification: line.quantification,group:line.group,quantification_type:line.quantification_type}]->(p);''',
                    "clinical":'''USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_proteomicsdata.csv" AS line 
                        MATCH (s:Sample {id:line.START_ID})
                        MATCH (c:Clinical_variable {id:line.END_ID)}) 
                        CREATE UNIQUE (s)-[:HAS_QUANTIFIED_CLINICAL{value: line.score}]->(c);''',
                    "project":'''CREATE CONSTRAINT ON (p:Project) ASSERT p.id IS UNIQUE; 
                        CREATE CONSTRAINT ON (p:Project) ASSERT p.name IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID.csv" AS line
                        MERGE (p:Project {id:line.ID}) 
                        ON CREATE SET p.name=line.name,p.description=line.description,p.responsible=line.responsible;
                        CREATE CONSTRAINT ON (s:Subject) ASSERT s.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_subjects.csv" AS line
                        MERGE (s:Sample {id:line.ID});
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file://IMPORTDIR/PROJECTID_project.csv" AS line 
                        MATCH (p:Project {id:line.START_ID})
                        MATCH (s:Sample {id:line.END_ID}) 
                        CREATE UNIQUE (p)-[:HAS_ENROLLED]->(s);
                        '''
                }
