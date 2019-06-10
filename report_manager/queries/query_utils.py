import ckg_utils

def read_queries(queries_file):
    queries = ckg_utils.get_queries(queries_file)

    return queries

def list_queries(queries):
    for q in queries:
        print("Name: {}\nDescription: {}\nInvolves nodes: {}\nInvolves relationships: {}\n Query: {}".format(queries[q]['name'],queries[q]['description'],",".join(queries[q]['involved_nodes']),",".join(queries[q]['involved_rels']),queries[q]['query']))

def find_queries_involving_nodes(queries, nodes):
    valid_queries = [ ]
    for q in queries:
        if len(set(queries[q]['involved_nodes']).intersection(nodes)) > 0:
            valid_queries.append(queries[q])

    return valid_queries

def find_queries_involving_relationships(queries, rels):
    valid_queries = []
    for q in queries:
        if len(set(queries[q]['involved_rels']).intersection(nodes)) > 0:
            valid_queries.append(queries[q])

    return valid_queries

def get_query(queries, query_id):
    query = None
    if query_id in queries:
        if "query" in queries[query_id]:
            query = queries[query_id]["query"]
    return query

def get_description(query):
    return query['description']

def get_nodes(query):
    return query['involved_nodes']

def get_relationships(query):
    return query['involved_rels']
