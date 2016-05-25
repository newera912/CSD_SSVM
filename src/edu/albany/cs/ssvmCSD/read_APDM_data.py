def read_APDM_data(path,flag=False):
    pvalue = {}
    graph = {}
    
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith('NodeID Weight'):
            n = idx + 1
            break
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            pvalue[int(items[0])]=float(items[1])
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            vertices = line.split(' ')
            edge = vertices[0]+'_'+vertices[1]
            cost = float(vertices[2])
            graph[edge] = cost
    #flag = False
    true_subgraph = []
    if flag == True:
        true_subgraph = []
        for idx in range(n, len(lines)):
            line = lines[idx]
            if line.find('END') >= 0:
                break
            else:
                items = line.split(' ')
                true_subgraph.append(int(items[0]))
                true_subgraph.append(int(items[1]))
        true_subgraph = sorted(list(set(true_subgraph)))
        return graph, pvalue, true_subgraph
    else:
        return graph,pvalue
def read_APDM_data_waterNetwork(path):
    pvalue = {}
    graph = {}
    
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith('NodeID Weight'):
            n = idx + 1
            break
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            pvalue[int(items[0])]=float(items[1])
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            vertices = line.split(' ')
            edge = vertices[0]+'_'+vertices[1]
            cost = float(vertices[2])
            graph[edge] = cost
    true_subgraph = []
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            break
        else:
            items = line.split(' ')
            true_subgraph.append(int(items[0]))
            true_subgraph.append(int(items[1]))
    true_subgraph = sorted(list(set(true_subgraph)))
    return graph, pvalue, true_subgraph

def read_APDM_data_transportation(path):
    pvalue = {}
    graph = {}
    
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.startswith('NodeID Weight Speed MeanSpeed'):
            n = idx + 1
            break
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            pvalue[int(items[0])]=float(items[1])
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            vertices = line.split(' ')
            edge = vertices[0]+'_'+vertices[1]
            cost = float(vertices[2])
            graph[edge] = cost
    return graph, pvalue


def read_APDM_data_CrimeOfChicago(path):
    b = {}
    c = {}
    graph = {}
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.startswith('NodeID PValue Counts'):
            n = idx + 1
            break
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            c[int(items[0])]=float(items[2].rstrip())
            b[int(items[0])]=float(items[3].rstrip())
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            vertices = line.split(' ')
            edge = vertices[0]+'_'+vertices[1]
            cost = float(vertices[2])
            graph[edge] = cost
    return graph, b,c