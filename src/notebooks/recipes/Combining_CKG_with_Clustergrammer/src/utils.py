import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn import metrics as sklm

def convert_to_numeric(data):
    df = data.copy()
    columns = df.columns
    df_new = pd.DataFrame(columns = columns)
    for i in columns:
        new_values = pd.to_numeric(df[i], errors = 'ignore')
        df_new[i] = new_values
    return df_new

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                
    return col_corr

def model_performance_cv (clf, X, y, features_selected, n_repeats):
    """
    Perform repeated cross-validation with random splits of the data.
    """

    performance_all = pd.DataFrame(columns = ['num_feat', 'train_roc_auc', 'test_roc_auc', 
                                          'features', 'precision',
                                          'sensitivity', 'specificity', 'F1-score', 'accuracy', 'MCC'])
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats = n_repeats, random_state=0)
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        performance = model_performance(clf = clf, features = features_selected, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
        performance_all = performance_all.append(performance).round(2)
    
    return performance_all

def _get_metrics(y_true, y_pred, cutoff=0.5):
    """Caculate a set of predefined binary metrics.

    Parameters:
    y: Pandas Series
        observed, true binary outcome
    y_pred_score: pandas.Series 
        Binary predictions from model
    cutoff: float
        Probablity cutoff for classification as positive.


    Return:
    metrics: dict, keys are the metrics computed
    """
    y_pred = y_pred >= cutoff
    sklearn_binary_metrics = ["roc_auc_score", 
                        "auc",
                        "cohen_kappa_score",
                        "matthews_corrcoef",
                        "f1_score",
                        "recall_score",
                        "precision_score",
                        "confusion_matrix"
                    ]
    metrics = {}
    for metric_key in sklearn_binary_metrics:
        metric_fct = getattr(sklm, metric_key)
        metric_value = metric_fct(y_true=y_true, y_pred=y_pred)
        metrics[metric_key] = metric_value
    
    # balanced accuracy
    # TN, FP, FN, TP unpacking
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = metrics["confusion_matrix"]    
    metrics["acc_bal"] = sklm.recall_score(y, y_pred_binary, average='macro')
    
    return  metrics

def model_performance(clf, features, X_train, y_train, X_test, y_test):
    clf.fit(X_train[features], y_train)

    pred_train = clf.predict_proba(X_train[features])
    pred_test = clf.predict_proba(X_test[features])
    
    # y_pred_train = clf.predict(X_train[features]) # not used
    y_pred_test = clf.predict(X_test[features])
    
    
    
    num_feat = len(features)

    metrics_train = _get_metrics(y_true=y_train, y_pred=pred_train, cutoff=0.5)
    metrics_test  = _get_metrics(y_true=y_test, y_pred=pred_test, cutoff=0.5)

    # train_roc_auc = sklm.roc_auc_score(y_train, pred_train[:,1])
    # test_roc_auc = sklm.roc_auc_score(y_test, pred_test[:,1])

    # precision = tp/(tp + fp) if (tp + fp)!= 0 else np.nan
    # sensitivity = tp/(tp + fn) if (tp + fn)!= 0 else np.nan
    # specificity = tn/(tn + fp) if (tn + fp)!= 0 else np.nan    
    # f1_score = 2*precision*sensitivity/(precision+sensitivity) if (precision + sensitivity) != 0 else np.nan    
    # accuracy = (tp + tn)/(tp + tn + fp + fn) if (tp + tn +fp +fn) !=0 else np.nan
    # mcc_a = tp * tn - fp * fn
    # mcc_b = np.sqrt((tp + fp) * (fn + tn) * (fp + tn) * (tp + fn))
    # mcc = mcc_a/mcc_b if mcc_b != 0 else np.nan
    # precision = sklm.precision_score(y_true=y_test, y_pred=y_pred_test)


    # metrics = [num_feat, train_roc_auc, test_roc_auc, 
    #           features, precision, sensitivity, specificity, f1_score, accuracy, mcc]
    result = pd.DataFrame(metrics).T
    result.columns = ['num_feat', 
                      'train_roc_auc', 
                      'test_roc_auc', 
                      'features',
                      'precision',
                      'sensitivity',
                      'specificity',
                      'F1-score',
                      'accuracy',
                      'MCC']
    return result

def feature_selection_by_rocauc(X_train, y_train, X_test, y_test, features):
    roc_values = []
    clf = clf_lr
    for feature in features:
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict_proba(X_test[feature].to_frame())
        roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

    score = pd.DataFrame(roc_values, columns = ['roc_auc_score'])
    score['features'] = features
    score['Gene names'] = score['features'].map(IDmapping_UniprotID_to_Genename)
    score = score.set_index('Gene names').sort_values(by = 'roc_auc_score', ascending = False)
    return (score)

def feature_selection(X, y, features, n_repeats):
    performances = []
    for feature in features:
        performance = model_performance_cv(X = X, y = y, features_selected= [feature], n_repeats = n_repeats).mean()
        columns = performance.index
        performances.append(list(performance))
    result = pd.DataFrame(performances, columns = columns)
    result['feature'] = features
    result['Gene name'] = result['feature'].map(IDmapping_UniprotID_to_Genename)
    return (result)

def feature_selection_bestcombo(features, X_train, y_train, n_repeats = 5):
    performances = []
    combo_features = []
    for k in range(6, 11):
        for i, j in enumerate(itertools.combinations(features, k)):
            combo = list(j)
            combo_features.append(combo)
            performance = model_performance_cv(X = X_train, y = y_train, features_selected = combo, n_repeats = n_repeats).mean()
            index = performance.index
            performance = list(performance)
            performances.append(performance)
    result = pd.DataFrame(performances, columns= index)
    result['features'] = combo_features
    return (result)

def plot_roc(clf, items, names, colors, X0, y0, X, y, title_1, title_2):
    fig, ax = plt.subplots(figsize = (4, 4))
    k = 0
    for item in items:
        name = names[k]
        selected_features = items[k]
        clf.fit(X0[selected_features], y0)
        pred = clf.predict_proba(X[selected_features])
        
        n_bootstraps = 1000
        rng_seed = 0
        scores = []
        roc_normal = roc_auc_score(y, pred[:, 1]).round(2)
        rng = np.random.RandomState(rng_seed)       
        
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the predcition indices
            indices = rng.randint(0, len(pred) -1, len(pred))
            if len(np.unique(y[indices])) < 2:
                # we need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            score = roc_auc_score(y[indices], pred[indices][:, 1])
            scores.append(score)
        score_mean = np.array(scores).mean().round(2)
        confidence_lower = sorted(scores)[int(0.025 * len(sorted(scores)))]
        confidence_upper = sorted(scores)[int(0.975 * len(sorted(scores)))]
        roc_auc = roc_auc_score(y, pred[:,1])
    
        fpr = dict()
        tpr = dict()
        roc_aucs = dict()
        for i in range(0, 2):
            fpr[i], tpr[i], thresholds = roc_curve(y, pred[:, i], pos_label=1)
            roc_aucs[i] = auc(fpr[i], tpr[i])       

        lw = 1
        plt.plot(fpr[1], tpr[1], color=colors[k], lw=lw, label='{}: AUC = {}'.format(name, roc_normal))
    
        plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False positive rate', fontsize=15)
        plt.ylabel('True positive rate', fontsize=15)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.legend(loc="lower right")
        k +=1
        plt.title('{}\n{}'.format(title_1, title_2), fontsize = 15)
        plt.savefig('figures/model/3C/{}_{}.png'.format(title_1, title_2),dpi = 120, bbox_inches = 'tight')