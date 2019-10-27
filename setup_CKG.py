import os

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    structure_directory = '.'
    data_directory_structure = {"data":[
                                    "archive",
                                    "databases",
                                    "experiments",
                                    "imports/databases",
                                    "imports/experiments",
                                    "imports/ontologies",
                                    "imports/stats",
                                    "ontologies"
                                    ]
                                }
    
    for directory in data_directory_structure:
        new_dir = os.path.join(os.path.join(path, structure_directory), directory)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for int_dir in data_directory_structure[directory]:
            new_int_dir = os.path.join(new_dir,int_dir)
            if not os.path.exists(new_int_dir):
                os.makedirs(new_int_dir)
