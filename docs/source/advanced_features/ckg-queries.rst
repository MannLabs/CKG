Retrieving data from the Clinical Knowledge Graph database
==========================================================

CKG has multiple cypher queries predefined to extract knowledge from the graph. These queries are part of:

- Graph Database Builder queries: define how to load data into the graph (see `graphdb_builder cypher.yml <https://raw.githubusercontent.com/MannLabs/CKG/master/ckg/graphdb_builder/builder/cypher.yml>`__)

- `Report manager queries <https://github.com/MannLabs/CKG/tree/master/ckg/report_manager/queries>`__: 
  - Dataset queries: extract data and knowledge for each data type integrated and analyzed in the graph for analysis with the analytics core (`dataset_cypher.yml <https://raw.githubusercontent.com/MannLabs/CKG/master/ckg/report_manager/queries/datasets_cypher.yml>`__)
  - Knowledge queries: 
    - Annotations: extract knowledge based on list of proteins or drugs (`knowledge_annotation.yml <https://raw.githubusercontent.com/MannLabs/CKG/master/ckg/report_manager/queries/knowledge_annotation.yml>`__)


These queries have been defined in YAML format. This structure allows assigning attributes such as name, description or involved nodes and relationships, which make the queries searchable using the functionality we implemented in `query_utils.py <https://github.com/MannLabs/CKG/blob/master/ckg/graphdb_connector/query_utils.py>`__.

For instance, we can use functions in this module to find within a query file, which queries involve specific types of nodes or relationships: 

.. code-block:: python

    selected_queries = {}
    queries = query_utils.read_queries(queries_file="ckg/report_manager/queries/knowledge_annotation.yml")
    for data_type in queries:
        selected_queries[data_type] = query_utils.find_queries_involving_nodes(queries=queries[data_type], nodes=["Protein", "Disease"], print_pretty=True)


If you want to contribute, you can add new queries following the same structure and they will then be available for everyone.

.. code-block:: yaml

    identifier:
        name: ... # string
        description: ... # string 
        involved_nodes: # list
          - ... # node type in the graph
          - ...
        involved_rels: # list
          - ... # relationship type in the graph
        query: > # string
            ...
            ...
            ... # Cypher query
    

