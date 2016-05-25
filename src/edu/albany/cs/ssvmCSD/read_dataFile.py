import read_APDM_data
import os
import os.path as path
import numpy as np

def getXY(filePath):
    #BASE_DIR = path.abspath(path.join(__file__ ,"../../../../../.."))
    #print BASE_DIR
    print "->",filePath
    graph,pvalue,true_subgraph=read_APDM_data.read_APDM_data(filePath,flag=True)    
    x=[]
    y=[0 for i in range(len(pvalue))]
    
    for k in pvalue:
        x.append(pvalue[k])
    for index in true_subgraph:
        y[index]=1
    print x
    print y
    return x,y



from os import listdir
from os.path import isfile, join
mypath=path.abspath(path.join(__file__ ,"../../../../../.."))+"/data/"
print mypath
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print onlyfiles