"""A module that SVM^python interacts with to do its evil bidding."""

# Thomas Finley, tfinley@gmail.com
x_to_file_dictionary=dict()


def read_examples(folder_path, sparm):
    """Reads and returns x,y example pairs from a file.    
    This reads the examples contained at the file at path filename and
    returns them as a sequence.  Each element of the sequence should
    be an object 'e' where e[0] and e[1] is the pattern (x) and label
    (y) respectively.  Specifically, the intention is that the element
    be a two-element tuple containing an x-y pair."""
 
    
    import read_dataFile
    from os import listdir
    from os.path import isfile, join
    
    examples=[]
    
    global x_to_file_dictionary
    
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for fileName in onlyfiles:
        file_path=folder_path+fileName
        x,y=read_dataFile.getXY(file_path)
        examples.append((x,y))
        x_to_file_dictionary[str(x)]=file_path
    return examples

def init_model(sample, sm, sparm):
    """Initializes the learning model.
    
    Initialize the structure model sm.  The sm.size_psi must be set to
    the number of features.  The ancillary purpose is to add any
    information to sm that is necessary from the user code
    perspective.  This function returns nothing."""
    # In our binary classification task, we've encoded a pattern as a
    # list of four features.  We just want a linear rule, so we have a
    # weight corresponding to each feature.  We also add one to allow
    # for a last "bias" feature.
    sm.size_psi = 6
    print 'feature num:',sm.size_psi



def classify_example(x, sm, sparm):
    """Given a pattern x, return the predicted label."""
    # Believe it or not, this is a dot product.  The last element of
    # sm.w is assumed to be the weight associated with the bias
    # feature as explained earlier.
    return sum([i*j for i,j in zip(x,sm.w[:-1])]) + sm.w[-1]

def find_most_violated_constraint(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.
    Returns the label ybar for pattern x corresponding to the most
    violated constraint according to SVM^struct cost function.  To
    find which cost function you should use, check sparm.loss_type for
    whether this is slack or margin rescaling (1 or 2 respectively),
    and check sparm.slack_norm for whether the slack vector is in an
    L1-norm or L2-norm in the QP (1 or 2 respectively).

    If this function is not implemented, this function is equivalent
    to 'classify(x, sm, sparm)'.  The optimality guarantees of
    Tsochantaridis et al. no longer hold, since this doesn't take the
    loss into account at all, but it isn't always a terrible
    approximation.  One still technically maintains the empirical
    risk bound condition, but without any regularization.
    score = classify_example(x,sm,sparm)
    discy, discny = y*score, -y*score + 1
    if discy > discny: return y
    return -y"""
    import os
    import os.path as path
    import jnius_config
    from jnius import autoclass
    global x_to_file_dictionary
    jnius_config.add_classpath(os.path.dirname(__file__)+"/*")
    print jnius_config.expand_classpath()
    
    
    
    System = autoclass('java.lang.System')
    print System.getProperty('java.class.path')
    
    IHT = autoclass('edu.albany.cs.ssvmCSD.IHT_Bridge')
    file_name=x_to_file_dictionary[str(x)]
    print 'constrain func:',file_name
    iht_ins = IHT(path.abspath(path.join(__file__ ,"../../../../../.."))+"/data/"+file_name)
    ybar=iht_ins.getX()
    print 'ybar:-',ybar
    return ybar
    

def find_most_violated_constraint_slack(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for slack rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def find_most_violated_constraint_margin(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for margin rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def psi(x, y, sm, sparm):
    """Return a feature vector representing pattern x and label y.

    This is the combined feature function, which this returns either a
    svmapi.Sparse object, or sequence of svmapi.Sparse objects (useful
    during kernel evaluations, as all components undergo kernel
    evaluation separately).  There is no default behavior."""
    # In the case of binary classification, psi is just the class (+1
    # or -1) times the feature vector for x, including that special
    # constant bias feature we pretend that we have.
#    1^T.y is subgraph size
#    x*y/1^t*y is the average feature value with the subgraph
#    x*1^T/n is the average feature value in the whole graph
#     is the average feature value outside the subgraph
#    the last two are the nonlinear transformations of the average feature value within the subgraph
    import svmapi
    import numpy as np
    SubGraph_size=svmapi.Sparse(sum(y),kernel_id=1)
    avg_SubGraph=svmapi.Sparse(np.dot(x,y),kernel_id=2)
    avg_WholeGraph=svmapi.Sparse(sum(x)/len(x),kernel_id=3)
    avg_OutSubgraph=svmapi.Sparse(np.dot(x,np.subtract(1,y))/sum(np.subtract(1,y)),kernel_id=4)
    avg_Quad_SubGraph=svmapi.Sparse(np.dot(x,y)*np.dot(x,y),kernel_id=5)
    avg_Log_SubGraph=svmapi.Sparse(np.log(np.dot(x,y)),kernel_id=6)
    
    psi=svmapi.Document([SubGraph_size,avg_SubGraph,avg_WholeGraph,avg_OutSubgraph,avg_Quad_SubGraph,avg_Log_SubGraph])
    print "psi called..........................."
    print psi
    return psi

def loss(y, ybar, sparm):
    """Return the loss of ybar relative to the true labeling y.
    
    Returns the loss for the correct label y and the predicted label
    ybar.  In the event that y and ybar are identical loss must be 0.
    Presumably as y and ybar grow more and more dissimilar the
    returned value will increase from that point.  sparm.loss_function
    holds the loss function option specified on the command line via
    the -l option.

    The default behavior is to perform 0/1 loss based on the truth of
    y==ybar."""
    # If they're the same sign, then the loss should be 0.
    #if y*ybar > 0: return 0
    print 'loss called.................................'
    from sklearn.metrics import hamming_loss    
    
    return hamming_loss(y, ybar)

def print_iteration_stats(ceps, cached_constraint, sample, sm,
                          cset, alpha, sparm):
    """Called just before the end of each cutting plane iteration.

    This is called just before the end of each cutting plane
    iteration, primarily to print statistics.  The 'ceps' argument is
    how much the most violated constraint was violated by.  The
    'cached_constraint' argument is true if this constraint was
    constructed from the cache.
    
    The default behavior is that nothing is printed."""
    print

def print_learning_stats(sample, sm, cset, alpha, sparm):
    """Print statistics once learning has finished.
    
    This is called after training primarily to compute and print any
    statistics regarding the learning (e.g., training error) of the
    model on the training sample.  You may also use it to make final
    changes to sm before it is written out to a file.  For example, if
    you defined any non-pickle-able attributes in sm, this is a good
    time to turn them into a pickle-able object before it is written
    out.  Also passed in is the set of constraints cset as a sequence
    of (left-hand-side, right-hand-side) two-element tuples, and an
    alpha of the same length holding the Lagrange multipliers for each
    constraint.

    The default behavior is that nothing is printed."""
    print 'Model learned:',
    print '[',', '.join(['%g'%i for i in sm.w]),']'
    print 'Losses:',
    print [loss(y, classify_example(x, sm, sparm), sparm) for x,y in sample]

def print_testing_stats(sample, sm, sparm, teststats):
    """Print statistics once classification has finished.
    
    This is called after all test predictions are made to allow the
    display of any summary statistics that have been accumulated in
    the teststats object through use of the eval_prediction function.

    The default behavior is that nothing is printed."""
    print teststats

def eval_prediction(exnum, (x, y), ypred, sm, sparm, teststats):
    """Accumulate statistics about a single training example.
    
    Allows accumulated statistics regarding how well the predicted
    label ypred for pattern x matches the true label y.  The first
    time this function is called teststats is None.  This function's
    return value will be passed along to the next call to
    eval_prediction.  After all test predictions are made, the last
    value returned will be passed along to print_testing_stats.

    On the first call, that is, when exnum==0, teststats==None.  The
    default behavior is that the function does nothing."""
    if exnum==0: teststats = []
    print 'on example',exnum,'predicted',ypred,'where correct is',y
    teststats.append(loss(y, ypred, sparm))
    return teststats

def write_model(filename, sm, sparm):
    """Dump the structmodel sm to a file.
    
    Write the structmodel sm to a file at path filename.

    The default behavior is equivalent to
    'cPickle.dump(sm,bz2.BZ2File(filename,'w'))'."""
    import cPickle, bz2
    f = bz2.BZ2File(filename, 'w')
    cPickle.dump(sm, f)
    f.close()

def read_model(filename, sparm):
    """Load the structure model from a file.
    
    Return the structmodel stored in the file at path filename, or
    None if the file could not be read for some reason.

    The default behavior is equivalent to
    'return cPickle.load(bz2.BZ2File(filename))'."""
    import cPickle, bz2
    return cPickle.load(bz2.BZ2File(filename))


