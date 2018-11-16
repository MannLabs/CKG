from report_manager.queries import datasets_cypher
from report_manager import analysisResult as ar, report as rp
from report_manager.analyses import basicAnalysis
from graphdb_connector import connector


class Dataset:
    def __init__(self, identifier, dtype, configuration, data, analyses):
        self.identifier = identifier
        self.type = dtype
        self.configuration = configuration
        self.data = data
        self.analyses = analyses
        if len(data) == 0:
            self.data = self.queryData()

    def getIdentifier(self):
        return self.identifier

    def getType(self):
        return self.type

    def getData(self):
        return self.data

    def getDataset(self, dataset):
        if dataset in self.data:
            return self.data[dataset]
        return None

    def getAnalyses(self):
        return self.analyses
    
    def getAnalysis(self, analysis):
        if analysis in self.analyses:
            return self.analyses[analysis]
        return None

    def updateData(self, new):
        self.data.update(new)

    def updateAnalyses(self, new):
        self.analyses.update(new)
    
    def getConfiguration(self):
        return self.configuration

    def setIdentifier(self, identifier):
        self.identifier = identifier

    def setType(self, dtype):
        self.type = dtype

    def setConfiguration(self, configuration):
        self.configuration  = configuration

    def setData(self, data):
        self.data = data

    def setAnalyses(self, analyses):
        self.analyses = analyses

    def queryData(self):
        data = {}
        driver = connector.getGraphDatabaseConnectionConfiguration()
        replace = [("PROJECTID", self.getIdentifier())]
        if "replace" in self.getConfiguration():
            replace = self.getConfiguration()["replace"]
        for query_name in datasets_cypher.queries[self.getType()]:
            title = query_name.lower().replace('_',' ')
            query = datasets_cypher.queries[self.getType()][query_name]
            for r,by in replace:
                query = query.replace(r,by)
            data[title] = connector.getCursorData(driver, query)
        return data

    def generateReport(self):
        report = rp.Report(self.getType().capitalize())
        for key in self.getConfiguration():
            for section_query,analysis_types,plot_names,args in self.getConfiguration()[key]:
                if section_query in self.getData():
                    data = self.getData()[section_query]
                    result = None 
                    if len(analysis_types) >= 1:
                        for analysis_type in analysis_types:
                            result = ar.AnalysisResult(self.getIdentifier(), analysis_type, args, data)
                            self.updateAnalyses(result.getResult())
                            if key == "regulation":
                                reg_data = result.getResult()[analysis_type]
                                if not reg_data.empty:
                                    sig_hits = list(set(reg_data.loc[reg_data.rejected,"identifier"]))
                                    #sig_names = list(set(reg_data.loc[reg_data.rejected,"name"]))
                                    sig_data = data[sig_hits]
                                    sig_data.index = data['group'].tolist()
                                    self.updateData({"regulated":sig_data})
                            for plot_name in plot_names:
                                plots = result.getPlot(plot_name, section_query+"_"+analysis_type+"_"+plot_name, analysis_type.capitalize())
                                report.updatePlots({(analysis_type,plot_name):plots})
                    else:
                        if result is None:
                            dictresult = {}
                            dictresult["_".join(section_query.split(' '))] = data
                            result = ar.AnalysisResult(self.getIdentifier(),"_".join(section_query.split(' ')), {}, data, result = dictresult)
                            self.updateAnalyses(result.getResult())
                        for plot_name in plot_names:
                            plots = result.getPlot(plot_name, "_".join(section_query.split(' '))+"_"+plot_name, section_query.capitalize())
                            report.updatePlots({("_".join(section_query.split(' ')),plot_name):plots})
        return report

class ProteomicsDataset(Dataset):
    def __init__(self, identifier, configuration, data={}, analyses={}):
        Dataset.__init__(self, identifier, "proteomics", configuration, data, analyses)
        self.preprocessDataset()
        
    def preprocessDataset(self):
        processed_data = self.preprocessing()
        self.updateData({"preprocessed":processed_data})
    
    def preprocessing(self):
        processed_data = None
        data = self.getDataset("dataset")
        if data is not None:
            imputation = True
            method = "mixed"
            missing_method = 'percentage'
            missing_max = 0.3
            value_col = 'LFQ intensity'
            args = {}
            if "args" in self.getConfiguration():
                args = self.getConfiguration()["args"] 
            if "imputation" in args:
                imputation = args["imputation"]
            if "imputation_method" in args:
                method = args["imputation_method"]
            if "missing_method" in args:
                missing_method = args["missing_method"]
            if "missing_max" in args:
                missing_max = args["missing_max"]
            if "value_col" in args:
                value_col = args["value_col"]
            
            processed_data = basicAnalysis.get_measurements_ready(data, imputation = imputation, method = method, missing_method = missing_method, missing_max = missing_max)
        return processed_data

class WESDataset(Dataset):
    def __init__(self, identifier, configuration, data={}):
        Dataset.__init__(identifier, "wes", configuration, data)
