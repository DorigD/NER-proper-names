def precision_recall_fscore(y_true, y_pred, average='binary'):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    return precision, recall, f1

def accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

def classification_report(y_true, y_pred, target_names=None):
    from sklearn.metrics import classification_report as sklearn_classification_report
    return sklearn_classification_report(y_true, y_pred, target_names=target_names)