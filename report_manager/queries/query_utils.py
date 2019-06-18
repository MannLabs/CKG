import os
import sys
import ckg_utils
from graphdb_connector import connector

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

def map_node_name_to_id(driver, node, value):
    query_name = 'map_node_name'
    cwd = os.path.abspath(os.path.dirname(__file__))
    queries_path = "project_cypher.yml"
    cypher = read_queries(os.path.join(cwd, queries_path))
    query = cypher[query_name]['query'].replace('NODE', node)
    identifier = connector.getCursorData(driver, query, parameters={'name':str(value)}).values[0][0]
    return identifier

