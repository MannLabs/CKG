CREATE_DB = "LOAD"

CREATE_USER = "CALL dbms.security.createUser(username, password, requirePasswordChange = False)"
ADD_ROLE_TO_USER = "CALL dbms.security.addRoleToUser(rolename, username)"

COUNT_RELATIONSHIPS = "MATCH (:ENTITY1)-[:RELATIONSHIP]->(:ENTITY2) return count(*) AS count;"
REMOVE_RELATIONSHIPS = "MATCH (:ENTITY1)-[r:RELATIONSHIP]->(:ENTITY2) delete r;"
REMOVE_NODE = "all apoc.periodic.iterate(\"MATCH (n:ENTITY) return n\", \"DETACH DELETE n\", {batchSize:1000}) yield batches, total return batches, total"

IMPORT_ONTOLOGY_DATA = '''CREATE INDEX ON :ENTITY(name);
                        CREATE CONSTRAINT ON (e:ENTITY) ASSERT e.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY.csv" AS line
                        MERGE (e:ENTITY {id:line.ID})
                        ON CREATE SET e.name=line.name,e.description=line.description,e.type=line.type,e.synonyms=SPLIT(line.synonyms,','); 
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY_has_parent.csv" AS line
                        MATCH (e1:ENTITY{id:line.START_ID})
                        MATCH (e2:ENTITY{id:line.END_ID}) 
                        MERGE (e1)-[:HAS_PARENT]->(e2);
                        '''

IMPORT_PROTEIN_DATA = '''CREATE INDEX ON :Protein(accession); 
                        CREATE CONSTRAINT ON (a:Protein) ASSERT a.id IS UNIQUE; 
                        CREATE CONSTRAINT ON (a:Protein) ASSERT a.accesion IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Protein.csv" AS line
                        MERGE (p:Protein {id:line.ID}) 
                        ON CREATE SET p.accession=line.accession,p.name=line.name,p.description=line.description,p.taxid=line.taxid,p.synonyms=SPLIT(line.synonyms,',');
                        CREATE CONSTRAINT ON (p:Peptide) ASSERT p.id IS UNIQUE;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Peptide.csv" AS line
                        MERGE (p:Peptide{id:line.ID}) 
                        ON CREATE SET p.type=line.type,p.unique=line.unique;
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_peptide_belongs_to_protein.csv" AS line 
                        MATCH (p1:Peptide {id:line.START_ID})
                        MATCH (p2:Protein {id:line.END_ID}) 
                        MERGE (p1)-[:BELONGS_TO_PROTEIN{source:line.source}]->(p2);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_gene_translated_into.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (g:Gene {id:line.END_ID})
                        MERGE (g)-[:TRANSLATED_INTO]->(p);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_transcript_translated_into.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (t:Transcript {id:line.END_ID})
                        MERGE (t)-[:TRANSLATED_INTO]->(p);
                        '''
IMPORT_MODIFIED_PROTEINS = '''
                        CREATE CONSTRAINT ON (m:Modified_protein) ASSERT m.id IS UNIQUE;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Modified_protein.csv" AS line
                        MERGE (m:Modified_protein {id:line.ID})
                        ON CREATE SET m.protein=line.protein,m.position=line.position,m.residue=line.residue,m.sequence_window=line.sequence_window;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/psp_modified_protein_has_modification.csv" AS line
                        MATCH (mp:Modified_protein {id:line.START_ID})
                        MATCH (m:Modification {id:line.END_ID}) 
                        MERGE (mp)-[:HAS_MODIFICATION]->(m);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/psp_protein_has_modified_site.csv" AS line 
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (mp:Modified_protein {id:line.END_ID}) 
                        MERGE (p)-[:HAS_MODIFIED_SITE]->(mp);
                        '''
IMPORT_PROTEIN_ANNOTATIONS = '''USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_cellular_component_associated_with.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (c:Cellular_component {id:line.END_ID})
                        MERGE (p)-[:ASSOCIATED_WITH{score:toFloat(line.score),source:line.source,evidence_type:line.evidence_type}]->(c);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_molecular_function_associated_with.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (f:Molecular_function {id:line.END_ID})
                        MERGE (p)-[:ASSOCIATED_WITH{score:toFloat(line.score),source:line.source,evidence_type:line.evidence_type}]->(f);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_biological_process_associated_with.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (b:Biological_process {id:line.END_ID})
                        MERGE (p)-[:ASSOCIATED_WITH{score:toFloat(line.score),source:line.source,evidence_type:line.evidence_type}]->(b);
                        '''

IMPORT_COMPLEXES = '''CREATE CONSTRAINT ON (c:Complex) ASSERT c.id IS UNIQUE; 
                        CREATE CONSTRAINT ON (c:Complex) ASSERT c.name IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Complex.csv" AS line
                        MERGE (c:Complex {id:line.ID}) 
                        ON CREATE SET c.name=line.name,c.organism=line.organism,c.source=line.source,c.synonyms=SPLIT(line.synonyms,',');
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_protein_is_subunit_of.csv" AS line
                        MATCH (p:Protein {id:line.START_ID})
                        MATCH (c:Complex {id:line.END_ID})
                        MERGE (p)-[:IS_SUBUNIT_OF{cell_lines:SPLIT(line.cell_lines,','),evidences:SPLIT(line.evidences,','),publication:line.publication,source:line.source}]->(c);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_biological_process_associated_with.csv" AS line
                        MATCH (c:Complex {id:line.START_ID})
                        MATCH (b:Biological_process {id:line.END_ID})
                        MERGE (c)-[:ASSOCIATED_WITH{evidence_type:line.evidence_type,score:toFloat(line.score),source:line.source}]->(b);
                    '''

IMPORT_MODIFIED_PROTEIN_ANNOTATIONS = '''USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_disease_associated_with.csv" AS line
                        MATCH (m:Modified_protein {id:line.START_ID})
                        MATCH (d:Disease {id:line.END_ID})
                        MERGE (m)-[:ASSOCIATED_WITH{score:toFloat(line.score),source:line.source,evidence_type:line.evidence_type,publications:line.publications}]->(d);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_biological_process_associated_with.csv" AS line
                        MATCH (m:Modified_protein {id:line.START_ID})
                        MATCH (b:Biological_process {id:line.END_ID})
                        MERGE (m)-[:ASSOCIATED_WITH{score:toFloat(line.score),source:line.source,evidence_type:line.evidence_type,publications:line.publications,action:line.action}]->(b);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_substrate_is_substrate_of.csv" AS line
                        MATCH (m:Modified_protein {id:line.START_ID})
                        MATCH (p:Protein {id:line.END_ID})
                        MERGE (m)-[:IS_SUBSTRATE_OF{score:toFloat(line.score),source:line.source,evidence_type:line.evidence_type}]->(p);
                        '''

IMPORT_GENE_DATA = '''CREATE CONSTRAINT ON (g:Gene) ASSERT g.id IS UNIQUE; 
                        CREATE CONSTRAINT ON (g:Gene) ASSERT g.name IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Gene.csv" AS line
                        MERGE (g:Gene {id:line.ID}) 
                        ON CREATE SET g.name=line.name,g.family=line.family,g.taxid=line.taxid,g.synonyms=SPLIT(line.synonyms,','); 
                        '''

IMPORT_TRANSCRIPT_DATA = '''CREATE CONSTRAINT ON (t:Transcript) ASSERT t.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Transcript.csv" AS line
                        MERGE (t:Transcript {id:line.ID}) 
                        ON CREATE SET t.name=line.name,t.class=line.class,t.taxid=line.taxid,t.assembly=line.assembly; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/refseq_located_in.csv" AS line
                        MATCH (t:Transcript {id:line.START_ID})
                        MATCH (c:Chromosome {id:line.END_ID})
                        MERGE (t)-[:LOCATED_IN{start:line.start,end:line.end,strand:line.strand}]->(c);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/refseq_transcribed_into.csv" AS line
                        MATCH (g:Gene {id:line.START_ID})
                        MATCH (t:Transcript {id:line.END_ID})
                        MERGE (g)-[:TRANSCRIBED_INTO]->(t);
                        '''

IMPORT_CHROMOSOME_DATA = '''CREATE CONSTRAINT ON (c:Chromosome) ASSERT c.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Chromosome.csv" AS line
                        MERGE (c:Chromosome {id:line.ID})
                        ON CREATE SET c.name=line.name,c.taxid=line.taxid;
                        '''

IMPORT_CURATED_PPI_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_interacts_with.csv" AS line
                        MATCH (p1:Protein {id:line.START_ID})
                        MATCH (p2:Protein {id:line.END_ID})
                        MERGE (p1)-[:CURATED_INTERACTS_WITH{score:toFloat(line.score),interaction_type:line.interaction_type,method:SPLIT(line.method,','),source:SPLIT(line.source,','),publications:SPLIT(line.publications,',')}]->(p2);'''
IMPORT_COMPILED_PPI_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_interacts_with.csv" AS line
                        MATCH (p1:Protein {id:line.START_ID})
                        MATCH (p2:Protein {id:line.END_ID})
                        MERGE (p1)-[:COMPILED_INTERACTS_WITH{score:toFloat(line.score),interaction_type:line.interaction_type,source:SPLIT(line.source,','),scores:SPLIT(line.scores,','),evidences:SPLIT(line.evidences,',')}]->(p2);'''

IMPORT_INTERNAL_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY1_ENTITY2_associated_with_integrated.csv" AS line
                        MATCH (p:ENTITY1 {id:line.START_ID})
                        MATCH (d:ENTITY2 {id:line.END_ID})
                        MERGE (p)-[:ASSOCIATED_WITH{score:toFloat(line.score),source:line.source,evidence_type:line.evidence_type}]->(d);
                        '''
CREATE_PUBLICATIONS = '''
                    CREATE CONSTRAINT ON (p:Publication) ASSERT p.id IS UNIQUE; 
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Publications.csv" AS line
                    MERGE (p:Publication{id:line.ID})
                    ON CREATE SET p.linkout=line.linkout,p.journal=line.journal_title,p.PMC_id=line.pmcid,p.year=toInt(line.year),p.volume=line.volume,p.issue=line.issue,p.page=line.page,p.DOI=line.doi;
                    '''
                    
IMPORT_MENTIONS =   '''
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY_Publication_mentioned_in_publication.csv" AS line
                    MATCH (p:ENTITY {id:line.END_ID})
                    MATCH (d:Publication {id:line.START_ID})
                    MERGE (p)-[:MENTIONED_IN_PUBLICATION]->(d);
                    '''

IMPORT_PUBLISHED_IN =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY_published_in_publication.csv" AS line
                        MATCH (g:ENTITY {id:line.START_ID})
                        MATCH (p:Publication {id:line.END_ID})
                        MERGE (g)-[:PUBLISHED_IN]->(p);
                    '''

IMPORT_DISEASE_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/ENTITY_RESOURCE_associated_with.csv" AS line
                        MATCH (e:ENTITY {id:line.START_ID})
                        MATCH (d:Disease {id:line.END_ID})
                        MERGE (e)-[:ASSOCIATED_WITH{score:toFloat(line.score),evidence_type:line.evidence_type,source:line.source}]->(d);
                        '''

IMPORT_DRUG_DATA = '''CREATE INDEX ON :Drug(name);
                        CREATE CONSTRAINT ON (d:Drug) ASSERT d.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Drug.csv" AS line
                        MERGE (d:Drug {id:line.ID})
                        ON CREATE SET d.name=line.name,d.description=line.description,d.indication=line.indication,d.synonyms=line.synonyms,d.mechanism_of_action=line.mechanism_of_action,d.metabolism=line.metabolism,d.pharmacodynamics=line.pharmacodynamics,d.prices=line.prices,d.route_of_elimination=line.route_of_elimination,d.toxicity=line.toxicity,d.absorption=line.absorption,d.half_life=line.half_life,d.groups=line.groups,d.experimental_properties=line.experimental_properties,d.Melting_Point=line.Melting_Point,d.Hydrophobicity=line.Hydrophobicity,d.Isoelectric_Point=line.Isoelectric_Point,d.Molecular_Weight=line.Molecular_Weight,d.Molecular_Formula=line.Molecular_Formula,d.Water_Solubility=line.Water_Solubility,d.Monoisotopic_Weight=line.Monoisotopic_Weight,d.Polar_Surface_Area_PSA=line.Polar_Surface_Area_PSA,d.Refractivity=line.Refractivity,d.Polarizability=line.Polarizability,d.Rotatable_Bond_Count=line.Rotatable_Bond_Count,d.H_Bond_Acceptor_Count=line.H_Bond_Acceptor_Count,d.H_Bond_Donor_Count=line.H_Bond_Donor_Count,d.pKa_strongest_acidic=line.pKa_strongest_acidic,d.pKa_strongest_basic=line.pKa_strongest_basic,d.Physiological_Charge=line.Physiological_Charge,d.Number_of_Rings=line.Number_of_Rings,d.Bioavailability=line.Bioavailability,d.Rule_of_Five=line.Rule_of_Five,d.Ghose_Filter=line.Ghose_Filter,d.MDDR_Like_Rule=line.MDDR_Like_Rule,d.caco2_Permeability=line.caco2_Permeability,d.pKa=line.pKa,d.Boiling_Point=line.Boiling_Point;
                        '''

IMPORT_CURATED_DRUG_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_targets.csv" AS line
                        MATCH (d:Drug {id:line.START_ID})
                        MATCH (g:Gene {id:line.END_ID})
                        MERGE (d)-[:CURATED_TARGETS{source:line.source,interaction_type:line.type,score:'curated'}]->(g);'''

IMPORT_COMPILED_DRUG_DATA =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_associated_with.csv" AS line
                        MATCH (d:Drug {id:line.START_ID})
                        MATCH (g:Gene {id:line.END_ID})
                        MERGE (d)-[:COMPILED_TARGETS{score:toFloat(line.score), source:line.source,interaction_type:line.interaction_type,scores:SPLIT(line.evidences,','),evidences:SPLIT(line.evidences,',')}]->(g);'''

IMPORT_PPI_ACTION = '''
                     USING PERIODIC COMMIT 10000
                     LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_protein_acts_on_protein.csv" AS line
                     MATCH (p1:Protein {id:line.START_ID})
                     MATCH (p2:Protein {id:line.END_ID})
                     MERGE (p1)-[:ACTS_ON{source:line.source,action:line.action,score:toFloat(line.score),directionality:line.directionality}]->(p2);
                    '''

IMPORT_DRUG_SIDE_EFFECTS =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_has_side_effect.csv" AS line
                        MATCH (d:Drug {id:line.START_ID})
                        MATCH (p:Phenotype {id:line.END_ID})
                        MERGE (d)-[:HAS_SIDE_EFFECT{source:line.source,original_side_effect_code:line.original_side_effect}]->(p);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_is_indicated_for.csv" AS line
                        MATCH (d:Drug {id:line.START_ID})
                        MATCH (p:Phenotype {id:line.END_ID})
                        MERGE (d)-[:IS_INDICATED_FOR{source:line.source,original_side_effect_code:line.original_side_effect,evidence:line.evidence}]->(p);
                        '''

IMPORT_DRUG_ACTS_ON =   '''
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/RESOURCE_drug_acts_on_protein.csv" AS line
                        MATCH (d:Drug {id:line.START_ID})
                        MATCH (p:Protein {id:line.END_ID})
                        MERGE (d)-[:ACTS_ON{source:line.source,action:line.action,score:toFloat(line.score),directionality:line.directionality}]->(p);
                        '''

IMPORT_PATHWAY_DATA = '''
                        CREATE CONSTRAINT ON (p:Pathway) ASSERT p.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Pathway.csv" AS line
                        MERGE (p:Pathway{id:line.ID})
                        ON CREATE SET p.name=line.name,p.source=line.source;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_protein_associated_with_pathway.csv" AS line
                        MATCH (p:Protein{id:line.START_ID})
                        MATCH (a:Pathway{id:line.END_ID}) 
                        MERGE (p)-[:ANNOTATED_IN_PATHWAY{linkout:line.linkout,source:line.source}]->(a);
                        '''
IMPORT_METABOLITE_DATA = '''
                        CREATE CONSTRAINT ON (m:Metabolite) ASSERT m.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Metabolite.csv" AS line
                        MERGE (m:Metabolite{id:line.ID})
                        ON CREATE SET m.name=line.name,m.synonyms=line.synonyms,m.description=line.description,m.direct_parent=line.direct_parent,m.kingdom=line.kingdom,m.class=line.class,m.super_class=line.super_class,m.sub_class=line.sub_class,m.chemical_formula=line.chemical_formula,m.average_molecular_weight=line.average_molecular_weight,m.monoisotopic_molecular_weight=line.monoisotopic_molecular_weight,m.chebi_id=line.chebi_id,m.pubchem_compound_id=line.pubchem_compound_id,m.food_id=line.food_id;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_annotated_in_pathway.csv" AS line
                        MATCH (m:Metabolite{id:line.START_ID})
                        MATCH (p:Pathway{id:line.END_ID}) 
                        MERGE (m)-[:ANNOTATED_IN_PATHWAY{source:line.source}]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_associated_with_protein.csv" AS line
                        MATCH (m:Metabolite{id:line.START_ID})
                        MATCH (p:Protein{id:line.END_ID}) 
                        MERGE (m)-[:ASSOCIATED_WITH]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_associated_with_disease.csv" AS line
                        MATCH (m:Metabolite{id:line.START_ID})
                        MATCH (d:Disease{id:line.END_ID}) 
                        MERGE (m)-[:ASSOCIATED_WITH]->(d);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_associated_with_tissue.csv" AS line
                        MATCH (m:Metabolite{id:line.START_ID})
                        MATCH (t:Tissue{id:line.END_ID}) 
                        MERGE (m)-[:ASSOCIATED_WITH]->(t);
                        '''

IMPORT_KNOWN_VARIANT_DATA = '''
                            CREATE CONSTRAINT ON (k:Known_variant) ASSERT k.id IS UNIQUE; 
                            USING PERIODIC COMMIT 10000
                            LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/Known_variant.csv" AS line
                            MERGE (k:Known_variant {id:line.ID})
                            ON CREATE SET k.alternative_names=line.alternative_names;
                            USING PERIODIC COMMIT 10000
                            LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_chromosome_known_variant_found_in_chromosome.csv" AS line
                            MATCH (k:Known_variant {id:line.START_ID})
                            MATCH (c:Chromosome {id:line.END_ID}) 
                            MERGE (k)-[:VARIANT_FOUND_IN_CHROMOSOME]->(c);
                            USING PERIODIC COMMIT 10000
                            LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_gene_known_variant_found_in_gene.csv" AS line
                            MATCH (k:Known_variant {id:line.START_ID})
                            MATCH (g:Gene {id:line.END_ID}) 
                            MERGE (k)-[:VARIANT_FOUND_IN_GENE]->(g);
                            USING PERIODIC COMMIT 10000
                            LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/uniprot_protein_known_variant_found_in_protein.csv" AS line
                            MATCH (k:Known_variant {id:line.START_ID})
                            MATCH (p:Protein {id:line.END_ID}) 
                            MERGE (k)-[:VARIANT_FOUND_IN_PROTEIN]->(p);
                            '''

                        
IMPORT_CLINICALLY_RELEVANT_VARIANT_DATA = '''
                                        CREATE CONSTRAINT ON (k:Clinically_relevant_variant) ASSERT k.id IS UNIQUE; 
                                        USING PERIODIC COMMIT 10000
                                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_Clinically_relevant_variant.csv" AS line
                                        MERGE (k:Clinically_relevant_variant {id:line.ID})
                                        ON CREATE SET k.alternative_names=line.alternative_names,k.chromosome=line.chromosome,k.position=toInt(line.position),k.reference=line.reference,k.alternative=line.alternative,k.effect=line.effect,k.oncogeneicity=line.oncogeneicity;
                                        USING PERIODIC COMMIT 10000
                                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_known_variant_is_clinically_relevant.csv" AS line
                                        MATCH (k:Known_variant {id:line.START_ID})
                                        MATCH (c:Clinically_relevant_variant {id:line.END_ID}) 
                                        MERGE (k)-[:VARIANT_IS_CLINICALLY_RELEVANT{source:line.source}]->(c);
                                        USING PERIODIC COMMIT 10000
                                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_targets_clinically_relevant_variant.csv" AS line
                                        MATCH (d:Drug {id:line.START_ID})
                                        MATCH (k:Clinically_relevant_variant {id:line.END_ID})
                                        MERGE (d)-[:TARGETS_CLINICALLY_RELEVANT_VARIANT{association:line.association, evidence:line.evidence, tumor:line.tumor, type:line.type, source:line.source}]->(k);
                                        USING PERIODIC COMMIT 10000
                                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/SOURCE_associated_with.csv" AS line
                                        MATCH (k:Clinically_relevant_variant {id:line.START_ID})
                                        MATCH (d:Disease {id:line.END_ID})
                                        MERGE (k)-[:ASSOCIATED_WITH{score:line.score,evidence_type:line.evidence_type,source:line.source,number_publications:line.number_publications}]->(d);
                                        '''

IMPORT_GWAS = '''
                CREATE CONSTRAINT ON (g:GWAS_study) ASSERT g.id IS UNIQUE; 
                USING PERIODIC COMMIT 10000
                LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/GWAS_study.csv" AS line
                MERGE (g:GWAS_study {id:line.ID})
                ON CREATE SET g.title=line.title,g.date=line.date,g.sample_size=line.sample_size,g.replication_size=line.replication_size,g.trait=line.trait;
                '''

IMPORT_DATASETS = {"clinical":'''USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_clinical_state.csv" AS line 
                        MATCH (s:Subject{id:line.START_ID})
                        MATCH (c:Clinical_variable{id:line.END_ID}) 
                        MERGE (s)-[:HAS_CLINICAL_STATE{value:line.value}]->(c);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_clinical_quant.csv" AS line 
                        MATCH (s:Biological_sample{id:line.START_ID})
                        MATCH (c:Clinical_variable{id:line.END_ID}) 
                        MERGE (s)-[:HAS_QUANTIFIED_CLINICAL{value:toFloat(line.value)}]->(c);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_disease.csv" AS line 
                        MATCH (s:Subject {id:line.START_ID})
                        MATCH (d:Disease {id:line.END_ID}) 
                        MERGE (s)-[:HAS_DISEASE]->(d);
                        CREATE CONSTRAINT ON (t:Timepoint) ASSERT t.id IS UNIQUE;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_timepoint.csv" AS line
                        MERGE (t:Timepoint {id:line.ID})
                        ON CREATE SET t.units=line.units;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_biological_sample_timepoint.csv" AS line
                        MATCH (b:Biological_sample {id:line.START_ID})
                        MATCH (t:Timepoint {id:line.END_ID}) 
                        MERGE (b)-[:SAMPLED_AT_TIMEPOINT{intervention:line.intervention}]->(t);
                        ''',
                    "proteomics":'''USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_proteins.csv" AS line
                        MATCH (s:Analytical_sample {id:line.START_ID}) 
                        MATCH (p:Protein{id:line.END_ID}) 
                        MERGE (s)-[:HAS_QUANTIFIED_PROTEIN{value:toFloat(line.value),intensity:toFloat(line.Intensity),qvalue:toFloat(line.Qvalue),score:toFloat(line.Score),proteinGroup:line.id}]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_subject_peptide.csv" AS line 
                        MATCH (s:Analytical_sample {id:line.START_ID})
                        MATCH (p:Peptide {id:line.END_ID}) 
                        MERGE (s)-[:HAS_QUANTIFIED_PEPTIDE{value:toFloat(line.value), score:toFloat(line.Score), proteinGroupId:line.Protein_group_IDs}]->(p);
                        CREATE CONSTRAINT ON (m:Modified_protein) ASSERT m.id IS UNIQUE;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_modifiedprotein.csv" AS line
                        MERGE (m:Modified_protein {id:line.ID}) 
                        ON CREATE SET m.protein=line.protein,m.position=line.position,m.residue=line.residue,m.sequence_window=line.sequence_window;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_modifiedprotein_modification.csv" AS line
                        MATCH (mp:Modified_protein {id:line.START_ID})
                        MATCH (p:Modification {id:line.END_ID}) 
                        MERGE (mp)-[:HAS_MODIFICATION]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_modifiedprotein_protein.csv" AS line 
                        MATCH (mp:Modified_protein {id:line.START_ID})
                        MATCH (p:Protein {id:line.END_ID}) 
                        MERGE (mp)-[:HAS_MODIFIED_SITE]->(p);
                        USING PERIODIC COMMIT 10000 
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_modifiedprotein_subject.csv" AS line
                        MATCH (s:Analytical_sample{id:line.START_ID})
                        MATCH (mp:Modified_protein{id:line.END_ID}) 
                        MERGE (s)-[:HAS_QUANTIFIED_MODIFIED_PROTEIN{value:toFloat(line.value),sequenceWindow:line.Sequence_window,score:line.Score,deltaScore:line.Delta_score,scoreLocalization:line.Score_for_localization,localizationProb:line.Localization_prob}]->(mp);
                        ''',
                    "wes":'''
                        CREATE CONSTRAINT ON (s:Somatic_mutation) ASSERT s.id IS UNIQUE; 
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_somatic_mutation.csv" AS line
                        MERGE (s:Somatic_mutation{id:line.ID})
                        ON CREATE SET s.region=line.region,s.function=line.function,s.alternative_names=SPLIT(line.alternative_names, ','),s.Xref=line.Xref,s.SIFT_score=line.SIFT_score,
                                    s.SIFT_pred=line.SIFT_pred,s.Polyphen2_HDIV_score=line.Polyphen2_HDIV_score,s.Polyphen2_HDIV_pred=line.Polyphen2_HDIV_pred,
                                    s.Polyphen2_HVAR_score=line.Polyphen2_HVAR_score,s.Polyphen2_HVAR_pred=line.Polyphen2_HVAR_pred,s.LRT_score=line.LRT_score,
                                    s.LRT_pred=line.LRT_pred,s.MutationTaster_score=line.MutationTaster_score,s.MutationTaster_pred=line.MutationTaster_pred,
                                    s.MutationAssessor_score=line.MutationAssessor_score,s.MutationAssessor_pred=line.MutationAssessor_pred,s.FATHMM_score=line.FATHMM_score,
                                    s.FATHMM_pred=line.FATHMM_pred,s.PROVEAN_score=line.PROVEAN_score,s.PROVEAN_pred=line.PROVEAN_pred,s.VEST3_score=line.VEST3_score,
                                    s.CADD_raw=line.CADD_raw,s.CLINSIG=line.CLINSIG,s.CLNDBN=line.CLNDBN,s.CLNACC=line.CLNACC,s.CLNDSDB=line.CLNDSDB,s.CLNDSDBID=line.CLNDSDBID,
                                    s.cosmic70=line.cosmic70,s.ICGC_Id=line.ICGC_Id,s.ICGC_Occurrence=line.ICGC_Occurrence;
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_somatic_mutation_known_variant.csv" AS line
                        MATCH (s:Somatic_mutation {id:line.START_ID}) 
                        MATCH (k:Known_variant {id:line.END_ID})
                        MERGE (s)-[:IS_A_KNOWN_VARIANT]->(k);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_somatic_mutation_gene.csv" AS line
                        MATCH (s:Somatic_mutation {id:line.START_ID}) 
                        MATCH (g:Gene {id:line.END_ID})
                        MERGE (s)-[:VARIANT_FOUND_IN_GENE]->(g);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_somatic_mutation_chromosome.csv" AS line
                        MATCH (s:Somatic_mutation {id:line.START_ID}) 
                        MATCH (c:Chromosome {id:line.END_ID})
                        MERGE (s)-[:VARIANT_FOUND_IN_CHROMOSOME]->(c);
                        USING PERIODIC COMMIT 10000
                        LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_somatic_mutation_sample.csv" AS line
                        MATCH (a:Analytical_sample {id:line.START_ID})
                        MATCH (s:Somatic_mutation {id:line.END_ID}) 
                        MERGE (a)-[:CONTAINS_MUTATION]->(s);
                        '''
                }

CREATE_PROJECT = '''CREATE CONSTRAINT ON (p:Project) ASSERT p.id IS UNIQUE;
                    CREATE CONSTRAINT ON (p:Project) ASSERT p.name IS UNIQUE; 
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID.csv" AS line
                    MERGE (p:Project {id:line.ID}) 
                    ON CREATE SET p.name=line.name,p.description=line.description,p.acronym=line.acronym,p.responsible=line.responsible;
                    '''
CREATE_SUBJECTS = '''CREATE CONSTRAINT ON (s:Subject) ASSERT s.id IS UNIQUE;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_subjects.csv" AS line
                    MERGE (s:Subject {id:line.ID}) 
                    ON CREATE SET s.external_id=line.external_id;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_project.csv" AS line 
                    MATCH (p:Project {id:line.START_ID})
                    MATCH (s:Subject {id:line.END_ID}) 
                    MERGE (p)-[:HAS_ENROLLED]->(s);
                    '''

CREATE_BIOSAMPLES = '''CREATE CONSTRAINT ON (s:Biological_sample) ASSERT s.id IS UNIQUE; 
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_biological_samples.csv" AS line
                    MERGE (s:Biological_sample {id:line.ID})
                    ON CREATE SET s.source=line.source,s.external_id=line.external_id,s.owner=line.owner,s.collection_date=line.collection_date,s.conservation_conditions=line.conservation_conditions,s.storage=line.storage,s.status=line.status,s.quantity=line.quantity,s.quantity_units=line.quantity_units;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_subject_biosample.csv" AS line 
                    MATCH (b:Biological_sample {id:line.START_ID}) 
                    MATCH (s:Subject {id:line.END_ID})
                    MERGE (b)-[:BELONGS_TO_SUBJECT]->(s);'''

CREATE_ANALYTICALSAMPLES = '''CREATE INDEX ON :Analytical_sample(group);
                    CREATE CONSTRAINT ON (s:Analytical_sample) ASSERT s.id IS UNIQUE; 
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_analytical_samples.csv" AS line
                    MERGE (s:Analytical_sample {id:line.ID})
                    ON CREATE SET s.collection_date=line.collection_date,s.conservation_conditions=line.conservation_conditions,s.storage=line.storage,s.status=line.status,s.quantity=line.quantity,s.quantity_units=line.quantity_units,s.group=line.group,s.secondary_group=line.secondary_group;
                    USING PERIODIC COMMIT 10000
                    LOAD CSV WITH HEADERS FROM "file:///IMPORTDIR/PROJECTID_biosample_analytical.csv" AS line
                    MATCH (s1:Biological_sample {id:line.START_ID})
                    MATCH (s2:Analytical_sample {id:line.END_ID}) 
                    MERGE (s1)-[:SPLITTED_INTO{quantity:line.quantity,quantity_units:line.quantity_units}]->(s2);'''

IMPORT_PROJECT = [CREATE_PROJECT, CREATE_SUBJECTS, CREATE_BIOSAMPLES, CREATE_ANALYTICALSAMPLES]
