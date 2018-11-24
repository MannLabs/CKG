import ckg_utils


def read_queries(queries_file):
    queries = ckg_utils.get_queries(queries_file)

    return queries

def list_queries(queries):
    for q in queries:
        print("Name: {}\nDescription: {}\nInvolves nodes: {}\nInvolves relationships: {}\n Query: {}".format(queries[q]['name'],queries[q]['description'],",".join(queries[q]['involves_nodes']),",".join(queries[q]['involves_rel']),queries[q]['query']))

def find_queries_involving_nodes(queries, nodes):
    valid_queries = []
    for q in queries:
        if len(set(queries[q]['involves_nodes']).intersection(nodes)) > 0:
            valid_queries.append(queries[q])

    return valid_queries

def find_queries_involving_relationships(queries, rels):
    valid_queries = []
    for q in queries:
        if len(set(queries[q]['involves_rels']).intersection(nodes)) > 0:
            valid_queries.append(queries[q])

    return valid_queries

def get_description(querie):
    pass

def get_nodes(querie):
    pass

def get_relationships(querie):
    pass
