import pandas as pd
import numpy as np
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
        call = rpacks.importr(package, signature_translation=True, suppress_messages=True)
    elif call == "package":
        call = rpacks.importr(designation, signature_translation=True, suppress_messages=True)
    return call

def R_matrix2Py_matrix(r_matrix, index, columns):
    matrix_class = r_matrix.rclass[0]

    if matrix_class == 'character':
        df = np.array(r_matrix)
        df.shape = (len(index), len(columns))
        df = pd.DataFrame(df)

    elif matrix_class == 'data.frame' or 'matrix':
        df = pd.DataFrame(pandas2ri.ri2py(r_matrix))
        if df.shape != (len(index), len(columns)):
            df.shape = (len(index), len(columns))
        else:
            pass

    df.index, df.columns = index, columns

    return df
