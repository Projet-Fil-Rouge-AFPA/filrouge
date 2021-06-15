import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_feature_importance(importance,names,model_type):
    """
    Plot feature importance

    Parameters
    ----------
    importance: model.feature_importances_
    names: x.columns
    model_type: 'Random Forest'
    """
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' Feature Importance')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def runGridSearchClassifiers(X, y, cv, models_list, parameters_list, output_predict = True, n_jobs=1, verbose=1):

    

    from sklearn.model_selection import cross_val_predict, GridSearchCV
    import types
    import warnings
    warnings.filterwarnings("ignore")
    result_list = []
    best_result = {}
    best_score = 0
    X_no_name = X
    y_no_name = y
    
    if len(models_list) != len(parameters_list):
        print('Error: models and parameters lists do not have the same length', len(models_list), len(parameters_list))
        return -1
    
    if isinstance(cv, types.GeneratorType):
        cv = list(cv)

    for model, parameters in zip(models_list,parameters_list):
        result = {}
        #cv_temp = cv
        clf = GridSearchCV(estimator=model, 
                            param_grid=parameters, 
                            scoring={'accuracy_score' : 'accuracy', 'f1_score' : 'f1_weighted',
                                    'balanced_accuracy_score' : 'balanced_accuracy', 
                                    'precision' : 'precision_weighted', 'recall' : 'recall_weighted'}, 
                            refit='f1_score',
                            cv=cv, n_jobs=n_jobs, verbose=verbose)
        clf.fit(X_no_name, y_no_name)
        result['best_estimator'] = clf.best_estimator_
        result['best_score'] = clf.best_score_
        result['best_params'] = clf.best_params_
        result['mean_test_f1_score'] = clf.cv_results_['mean_test_f1_score'][clf.best_index_]
        result['std_test_f1_score'] = clf.cv_results_['std_test_f1_score'][clf.best_index_]
        result['mean_test_accuracy_score'] = clf.cv_results_['mean_test_accuracy_score'][clf.best_index_]
        result['std_test_accuracy_score'] = clf.cv_results_['std_test_accuracy_score'][clf.best_index_]
        result['mean_test_balanced_accuracy_score'] = clf.cv_results_['mean_test_balanced_accuracy_score'][clf.best_index_]
        result['std_test_balanced_accuracy_score'] = clf.cv_results_['std_test_balanced_accuracy_score'][clf.best_index_]
        result['mean_test_precision'] = clf.cv_results_['mean_test_precision'][clf.best_index_]
        result['std_test_precision'] = clf.cv_results_['std_test_precision'][clf.best_index_]
        result['mean_test_recall'] = clf.cv_results_['mean_test_recall'][clf.best_index_]
        result['std_test_recall'] = clf.cv_results_['std_test_recall'][clf.best_index_]
        result_list.append(result)
        if result['best_score'] > best_score:
            best_score = result['best_score']
            best_result = result

        if verbose:
            print('Best estimator', clf.best_estimator_)
            print('Best results', clf.best_score_)
            print('Best params', clf.best_params_)
            print('accuracy (mean, std)', clf.cv_results_['mean_test_accuracy_score'][clf.best_index_], 
                    clf.cv_results_['std_test_accuracy_score'][clf.best_index_])
            print('f1 (mean, std)', clf.cv_results_['mean_test_f1_score'][clf.best_index_], 
                    clf.cv_results_['std_test_f1_score'][clf.best_index_])
            print('balanced accuracy (mean, std)', clf.cv_results_['mean_test_balanced_accuracy_score'][clf.best_index_],                               clf.cv_results_['std_test_balanced_accuracy_score'][clf.best_index_])
            print('precision (mean, std)', clf.cv_results_['mean_test_precision'][clf.best_index_], 
                    clf.cv_results_['std_test_precision'][clf.best_index_])
            print('recall (mean, std)', clf.cv_results_['mean_test_recall'][clf.best_index_], 
                    clf.cv_results_['std_test_recall'][clf.best_index_])
            print()
    
    if output_predict:
        y_predict = cross_val_predict(best_result['best_estimator'],X_no_name,y_no_name,cv=cv)
    else:
        y_predict = None

    return best_result,  y_predict, result_list
