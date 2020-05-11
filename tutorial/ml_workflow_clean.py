# clean ml_workflow code

import pydra
import matplotlib.pyplot as plt
import sklearn
import typing as ty



# load data
from sklearn import datasets
X, y = datasets.load_digits(n_class=10, return_X_y=True)


##############################################################################
# case 1 - hyperparameter tuning


# task 1 - splitting data
@pydra.mark.task
@pydra.mark.annotate({"return": {'X_tr': ty.Any, 'X_tt': ty.Any,
                                 'y_tr': ty.Any, 'y_tt': ty.Any}})
def simple_split(train_size, random_state=0):
    from sklearn.model_selection import train_test_split
    X, y = datasets.load_digits(n_class=10, return_X_y=True)
    X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, train_size=train_size, random_state=random_state)
    return(X_tr, X_tt, y_tr, y_tt)



# task 2 - fit a SVM and returns accuracy score
@pydra.mark.task
@pydra.mark.annotate({"return": {"score": ty.Any}})
def fit_SVM1(X_tr, y_tr, X_tt, y_tt, C=1, kernel='linear', gamma=1):
    from sklearn.svm import SVC
    clf = SVC(C=C, kernel=kernel, gamma=gamma)
    clf.fit(X_tr, y_tr)
    
    from sklearn.metrics import accuracy_score
    y_pred = clf.predict(X_tt)
    return(accuracy_score(y_tt, y_pred))



# inputs
wf1_inputs = {'train_size': [0.8, 0.5],
              'C': [0.1, 10], 
              'gamma': [1, 0.01], 
              'kernel': ['linear', 'rbf']}

# workflow
wf1 = pydra.Workflow(name="svm", 
                     input_spec=list(wf1_inputs.keys()), **wf1_inputs)

wf1.split(['train_size', 'gamma', 'C', 'kernel'])
wf1.add(simple_split(name='split_data', 
                     train_size=wf1.lzin.train_size))  

wf1.add(fit_SVM1(name='fit_clf', 
                 X_tr=wf1.split_data.lzout.X_tr, y_tr=wf1.split_data.lzout.y_tr, 
                 X_tt=wf1.split_data.lzout.X_tt, y_tt=wf1.split_data.lzout.y_tt,
                 kernel=wf1.lzin.kernel, C=wf1.lzin.C, gamma=wf1.lzin.gamma))  

wf1.set_output([("acc_score", wf1.fit_clf.lzout.score)])

with pydra.Submitter(plugin="cf") as sub:
    sub(wf1)
    
wf1.result()
wf1.result(return_inputs=True)

##############################################################################
# case 2 - comparing different metrics

# task 1 - reuses one in first case
# task 2 - fits a SVM then returns a trained model
@pydra.mark.task
@pydra.mark.annotate({"return": {"clf": ty.Any}})
def fit_SVM2(X_tr, y_tr, C=1, kernel='linear', gamma=1):
    from sklearn.svm import SVC
    clf = SVC(C=C, kernel=kernel, gamma=gamma)
    clf.fit(X_tr, y_tr)
    return(clf)


# task 3 - calculate metrics
@pydra.mark.task
@pydra.mark.annotate({"return": {"metric": ty.Any}})
def metric_score(clf, X_tt, y_tt, method):
    
    y_pred = clf.predict(X_tt)
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    if method == 'acc':
        metric = accuracy_score(y_tt, y_pred)
    elif method == 'f1':
        metric = f1_score(y_tt, y_pred, average='macro')
    else:
        metric = confusion_matrix(y_tt, y_pred)
    return(metric)


wf2_inputs = {'train_size': [0.5, 0.8],
              'C': [0.1, 10,], 
              'gamma': [1, 0.01], 
              'kernel': ['linear', 'rbf'],
              'metric': ['acc', 'f1']}


# workflow
wf2 = pydra.Workflow(name="svm2", 
                     input_spec=list(wf2_inputs.keys()), **wf2_inputs)

# Workflow-level splits
wf2.split(['train_size', 'kernel', 'gamma', 'C'])
wf2.add(simple_split(name='split_data', 
                     train_size=wf2.lzin.train_size))  

wf2.add(fit_SVM2(name='fit_clf', 
                X_tr=wf2.split_data.lzout.X_tr, y_tr=wf2.split_data.lzout.y_tr, 
                kernel=wf2.lzin.kernel, C=wf2.lzin.C, gamma=wf2.lzin.gamma))  

# Node-level split
wf2.add(metric_score(name='calc_metric', 
                     clf=wf2.fit_clf.lzout.clf,
                     X_tt=wf2.split_data.lzout.X_tt, y_tt=wf2.split_data.lzout.y_tt,
                     method=wf2.lzin.metric).split('method'))  # note the variable is called method

wf2.set_output([("metric", wf2.calc_metric.lzout.metric)])

with pydra.Submitter(plugin="cf") as sub:
    sub(wf2)

wf2.result(return_inputs=True)