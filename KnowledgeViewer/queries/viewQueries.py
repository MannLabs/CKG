PROJECT_OVERVIEW = {"Project details": 
                            '''MATCH (p:Project{id:"PROJECTID"}) RETURN p.name, p.id, p.description, p.responsible, p.status, p.participants, p.types;''',
                    "Number of Enrolled Subjects:":
                            '''MATCH (p:Project{id:"PROJECTID"})-[:HAS_ENROLLED]-(s:Subject) RETURN COUNT(DISTINCT(s));''',
                    "Number of Analytical Samples:":
                            '''MATCH (p:Project{id:"PROJECTID"})-[*3]-(a:Analytical_sample) RETURN COUNT(DISTINCT(a));''',
                    }

                    
