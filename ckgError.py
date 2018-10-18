class Error(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)
        
class InputError(Error):
        def __init__(self, value):
            self.value = "Input error: " + value

class WrongConfiguration(Error):
        def __init__(self, value):
            self.value = "Wrong configuration: " + value

class MissingConfiguration(Error):
        def __init__(self, missing_configuration):
            self.value = "Missing configuration: " + ", ".join(missing_configuration)

class DatabaseError(Error):
        def __init__(self, value):
            self.value = "Database - " + value

class ClientError(Error):
        def __init__(self, value):
            self.value = "Database client - " + value

class GraphError(Error):
        def __init__(self, value):
            self.value = "Database graph - " + value

class TransientError(Error):
        def __init__(self, value):
            self.value = "Database transient - " + value

class MappingError(Error):
        def __init__(self, value):
            self.value = "Mapping - " + value

class EmptyNetworkError(Error):
        def __init__(self, value):
            self.value = "Empty network - " + value

class PartnerURLNotAvailable(Error):
        def __init__(self, error, value):
            self.value = value +": "+ str(error).split('.')[0]

class ErrorReadingExperimentFile(Error):
        def __init__(self, error):
            self.value = "Error reading file: Please, check the format of the file."


