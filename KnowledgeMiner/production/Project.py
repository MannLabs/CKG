

class Project:
    def __init__(identifier, project_type, datasets = []):
        self.identifier = identifier
        self.project_type = project_type
        self.datasets = datasets

    def getIdentifier(self):
        return self.identifier

    def getProject_type(self):
        return self.project_type

    def getDatasets(self):
        return self.datasets

    def setIdentifier(self, identifier):
        self.identifier = identifier

    def setProject_type(self, project_type):
        self.project_type = project_type

    def setDatasets(self, datasets):
        self.datasets = datasets

    def generateReport(self):
        pass

