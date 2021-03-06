import os
import os.path as path
import jnius_config

jnius_config.add_classpath(os.path.dirname(__file__)+"/*")
print jnius_config.expand_classpath()

from jnius import autoclass

System = autoclass('java.lang.System')
print System.getProperty('java.class.path')

IHT = autoclass('edu.albany.cs.ssvmCSD.IHT_Bridge')
iht_ins = IHT(path.abspath(path.join(__file__ ,"../../../../../.."))+"/data/APDM-GridData-100-precen_0.05-noise_0-numCC_1_0.txt")
print iht_ins.getX()
