from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, FloatVector
import rpy2.robjects.packages as rpacks
pandas2ri.activate()

R = ro.r


def call_Rpackage(call="function", designation="aov"):
    if call == "function":
        package = rpacks.wherefrom(designation).do_slot("name")[0].split(":")[1]
        call = rpacks.importr(package)
    elif call == "package":
        call = rpacks.importr(designation)
    return call

def R_matrix2Py_matrix(r_matrix):
    df = pd.DataFrame(pandas2ri.ri2py(r_matrix))
    index, cols = R.rownames(r_matrix), R.colnames(r_matrix)
    df.index, df.columns = index, cols
    return df

def blockwiseModules(r, power, minModuleSize, reassignThreshold, mergeCutHeight, numericLabels, pamRespectsDendro, saveTOMs, saveTOMFileBase, verbose):
    function = R(''' net <- function(r, power, minModuleSize, reassignThreshold, mergeCutHeight, numericLabels, pamRespectsDendro,
                                    saveTOMs, saveTOMFileBase, verbose) {
                                    blockwiseModules(r, power=power, minModuleSize=minModuleSize, reassignThreshold = reassignThreshold,
                                    mergeCutHeight = mergeCutHeight, numericLabels = numericLabels, pamRespectsDendro = pamRespectsDendro,
                                    saveTOMs = saveTOMs, saveTOMFileBase = saveTOMFileBase, verbose = verbose)}''')
    return function

def paste_matrices(matrix1, sigint1, matrix2, sigint2):
    function = R(''' text <- function(matrix1, sigint1, matrix2, sigint2) {
                                    paste(signif(matrix1, sigint1), signif(matrix2, sigint2), sep = "<br>")} ''')
    return function
