#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#pipeline for classification

#Load the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
#Function to replace inf values with NaN
def replace_inf_with_nans(df, column):
    inf_indices_list = df.index[df[column] == float('inf')].tolist() #The inf are floats not objecttype str
    for i in range(len(inf_indices_list)):
        df[column][inf_indices_list[i]] = np.NaN
    return(df)



def replace_with_ceiling(df):
    RTindices_list = []
    GAZERXindices_list = []
    GAZERYindices_list = []
    GAZELXindices_list = []
    GAZELYindices_list = []

    conditioned2_df = df.loc[ (df['rt'] >=3000)]  #this the df with the (violated) condition
    GAZERXindices_list.append(df.loc[(df['right_gaze_x'] >= 10)].index)
    GAZERYindices_list.append(df.loc[(df['right_gaze_y']>=10)].index)
    GAZELXindices_list.append(df.loc[(df['left_gaze_x'] >= 10)].index)
    GAZELYindices_list.append(df.loc[(df['left_gaze_y'] >=10)].index)
    RTindices_list.append(conditioned2_df.index)
    import itertools
    
    RTind_list =list(itertools.chain(*RTindices_list))
    GAZERXind_list =list(itertools.chain(*GAZERXindices_list))
    GAZERYind_list =list(itertools.chain(*GAZERYindices_list))
    GAZELXind_list =list(itertools.chain(*GAZELXindices_list))
    GAZELYind_list =list(itertools.chain(*GAZELYindices_list))
    
    #Now replace these with nan in the right column
    df.loc[RTind_list, 'rt']= 3000   
    df.loc[GAZERXind_list, 'right_gaze_x']= 10
    df.loc[GAZERYind_list, 'right_gaze_y']= 10 
    df.loc[GAZELXind_list, 'left_gaze_x']= 10 
    df.loc[GAZELYind_list, 'left_gaze_y']= 10 
    return df


def replace_with_nan(df):
    RTindices_list = []
    GAZERXindices_list = []
    GAZERYindices_list = []
    GAZELXindices_list = []
    GAZELYindices_list = []
    conditioned2_df = df.loc[ (df['rt'] >=3000)]  #this the df with the (violated) condition
    GAZERXindices_list.append(df.loc[(df['right_gaze_x'] >= 10)].index)
    GAZERYindices_list.append(df.loc[(df['right_gaze_y']>=10)].index)
    GAZELXindices_list.append(df.loc[(df['left_gaze_x'] >= 10)].index)
    GAZELYindices_list.append(df.loc[(df['left_gaze_y'] >=10)].index)
    RTindices_list.append(conditioned2_df.index)
    import itertools
    
    RTind_list =list(itertools.chain(*RTindices_list))
    GAZERXind_list =list(itertools.chain(*GAZERXindices_list))
    GAZERYind_list =list(itertools.chain(*GAZERYindices_list))
    GAZELXind_list =list(itertools.chain(*GAZELXindices_list))
    GAZELYind_list =list(itertools.chain(*GAZELYindices_list))
    
    #Now replace these with nan in the right column
    df.loc[RTind_list, 'rt']= np.NaN  
    df.loc[GAZERXind_list, 'right_gaze_x']= np.NaN
    df.loc[GAZERYind_list, 'right_gaze_y']= np.NaN
    df.loc[GAZELXind_list, 'left_gaze_x']= np.NaN
    df.loc[GAZELYind_list, 'left_gaze_y']= np.NaN
    return df

'''
#Function to replace values with nan
def replace_with_nan(df):
    indices_list = []
    conditioned1_df = df.loc[(df['rt'] < 200)] #this is the df with (violated) condition
    conditioned2_df = df.loc[ (df['rt'] > 3000)]  #this the df with the (violated) condition
    indices_list.append(conditioned1_df.index)
    indices_list.append(conditioned2_df.index)
    import itertools
    ind_list =list(itertools.chain(*indices_list))
    #Now replace these with nan in the right column
    df.loc[ind_list, 'rt']= np.NaN      
    return df
'''

#impute at subject level  
def impute_per_subject(df):
    imputed_df_list = []
    subjects = list(df['subject'].unique())
    for i in range(len(subjects)):
        subject_df = df.loc[df['subject']== subjects[i]]
        subject_df_cols = subject_df.columns
        imputer = KNNImputer(n_neighbors=2)
        imputed_array = imputer.fit_transform(subject_df)
        imputed_subject_df = pd.DataFrame(data = imputed_array, columns = subject_df_cols)
        imputed_df_list.append(imputed_subject_df)
    imputed_df = pd.concat(imputed_df_list)
    return imputed_df


#Function to standardize certain columns of the df per subject
def standadize_cols_per_subject(df):
    scaler = MinMaxScaler()
    subjects = df['subject'].unique()
    list_of_subject_dfs = []
    for i in range(len(subjects)):
        subject_df = df.loc[df['subject'] == subjects[i]] 
        columns = subject_df.drop(['subject','trial'], axis =1).columns #the columns in the above df that needs to be standardized 
        for j in range(len(columns)): #standardize each column
            subject_df[columns[j]] = scaler.fit_transform(subject_df[columns[j]].values.reshape(-1,1))
        list_of_subject_dfs.append(subject_df)
    new_df = pd.concat(list_of_subject_dfs)
    return new_df



#get the feature matrix 349*980
def get_feature_matrix(df):

    subjects = df['subject'].unique()
    Feature_matrix= pd.DataFrame()#index = subjects)
    trials = df['trial'].unique()
    feature_df_names = ['mean_DIAM', 'var_DIAM', 'mean_LDEV', 'var_LDEV', 'mean_RDEV', 'var_RDEV', 'RT']
    feature_df_dict = dict(('df_' + str(x), pd.DataFrame()) for x in feature_df_names)
    df_cols = df.drop(['subject', 'trial'], axis =1).columns #the cols that provide features

    for i in range(len(subjects)):
        subject_df = df.loc[df['subject'] == subjects[i]]
        for j in range(len(trials)):
            trial = trials[j]
            subject_trial_df = subject_df.loc[subject_df['trial']== trial]
            for k in range(len(feature_df_names )):
                list(feature_df_dict.values())[k][str(trial)]= subject_trial_df[df_cols[k]].tolist()

        Feature_matrix = Feature_matrix.append(pd.concat(list(feature_df_dict.values()), axis=1) ) 

    #renaming the columns
    Feature_matrix.columns = list(range(1,981))
    #renaming the index
    Feature_matrix.index = subjects
    return Feature_matrix




#Add the labels back to the feature matrix

def get_feature_matrix_with_labels(df, df1):
    
    subjects = df['subject'].unique()

    bi_goal_list = []
    bi_feedback_list = []
    bi_reward_list = []
    tir_goal_list = []
    tir_reward_list = []
    label_7_list = []
    label_names = ['bi_goal', 'bi_feedback', 'bi_reward', 'tir_goal', 'tir_reward', 'label_7']
    label_list = [bi_goal_list, bi_feedback_list, bi_reward_list, tir_goal_list,tir_reward_list,label_7_list]

    for i in range (len(subjects)):
        subject_df = df.loc[df['subject'] == subjects[i]].head(1)
        for j in range(6):
            label_list[j].append(subject_df.iloc[0][label_names[j]])

    for i in range(6):
        df1[label_names[i]] = label_list[i]

    return df1



#compute the no.of n features that need to be used for each problem and the most frequent n features per problem
def get_frequent_features(df):

    #Feature selection

    #First getting the alpha values for LASSO and ELATICNET
    label_list = ['bi_goal','bi_feedback', 'bi_reward', 'tir_goal', 'tir_reward' , 'label_7']
    lasso_alphas = []
    elastic_alphas = []
    feature_df = df.iloc[:, 0:980]
    for i in range(6):
        label_series = df[label_list[i]]
        problem_df1 = pd.concat([feature_df, label_series], axis =1) 
        problem_df2 = problem_df1.dropna() #This is the correct df for the problem with the right subjects
        X = problem_df2.iloc[:, 0:980]
        X= X.astype('float')
        Y = problem_df2[label_list[i]]
        Y= Y.astype('int')
        #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        lassocv = LassoCV(alphas = None, cv = None, max_iter = 100000, random_state=0).fit(X, Y)
        elasticv = ElasticNetCV(alphas = None, cv = None, max_iter = 100000, random_state=0).fit(X, Y)
        lasso_alphas.append(lassocv.alpha_)
        elastic_alphas.append(elasticv.alpha_)


    #Getting the top features by a combination of LASSO and ElasticNet  
    #these will be the important features for each classification problem 
    bi_goal_list = []
    bi_feedback_list = []
    bi_reward_list = []
    tir_goal_list = []
    tir_reward_list = []
    label_7_list = []
    label_list =  ['bi_goal','bi_feedback', 'bi_reward', 'tir_goal', 'tir_reward' , 'label_7']  

    for j in range(2):
        alpha_list = [lasso_alphas, elastic_alphas]
        classifiers = [linear_model.Lasso, ElasticNet]
        for m in range(75):
            top_features_per_problem = []
            feature_df = df.iloc[:, 0:980]
            for i in range(6):
                label_series = df[label_list[i]]
                problem_df1 = pd.concat([feature_df, label_series], axis =1) 
                problem_df2 = problem_df1.dropna() #This is the correct df for the problem with the right subjects
                X = problem_df2.iloc[:, 0:980]
                X= X.astype('float')
                Y = problem_df2[label_list[i]]
                Y= Y.astype('int')
                           
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
                alphas = alpha_list[j][i] 
                clf = classifiers[j](alpha = alphas)
                
                cv_results = cross_validate(clf, X_train, Y_train, cv=25, return_estimator=True, scoring= 'r2')
                data = cv_results['test_score']
                median_acc = np.argsort(data)[len(data)//2]
                a = cv_results['estimator'][median_acc].coef_
                feature_support = [x for x in X.columns.tolist() if abs(a[x-1]) > 0 ] #this is the support in numbers. (x-1) since we start counting the columns at 1
                top_features_per_problem.append(feature_support)

            #update the lists
            bi_goal_list.append(top_features_per_problem[0])
            bi_feedback_list.append(top_features_per_problem[1])
            bi_reward_list.append(top_features_per_problem[2]) 
            tir_goal_list.append(top_features_per_problem[3])
            tir_reward_list.append(top_features_per_problem[4])        
            label_7_list.append(top_features_per_problem[5]) 


    #Getting the avearge number of key features
    the_list = [bi_goal_list, bi_feedback_list,  bi_reward_list, tir_goal_list, tir_reward_list, label_7_list]
    av_lasso = []
    av_elasti = []

    for i in range(6):
        current_list = the_list[i]
        lasso_list = current_list[0:75] 
        elasti_list = current_list[75:150] 
        av_lasso.append(sum([len(x) for x in lasso_list])/75)
        av_elasti.append(sum([len(x) for x in elasti_list])/75)

    av_list = [(g + h) / 2 for g, h in zip(av_lasso, av_elasti)] #this is the avearge number of features used in each problem

    #save the most frequent features for each problem in one list

    import itertools

    frequent_feature_list = []
    round_to_whole_av_list = [round(num) for num in av_list]

    for i in range(6):

        regular_list = the_list[i]
        flat_list = list(itertools.chain(*regular_list))
        dict_a = dict((x,flat_list.count(x)) for x in set(flat_list)) #counts the occurence of each feature
        sort_dict = sorted( dict_a.items(), key=lambda x: x[1], reverse=True) #sorting the above dictionary by value
        frequent_feature_list.append([j[0] for j in sort_dict[0:round_to_whole_av_list[i]+15]]) #most frequent features (in order)
        #we add 15 more features than suggested by LASSO+Elasticnet to check on the Elbow curve.
        
    return frequent_feature_list, round_to_whole_av_list


#get the optimal features for elbow curve
def get_stats_for_elbow_curve(df, feature_list):
    #random_state= 42 #For reproducability
    from sklearn.model_selection import train_test_split
    balanced_accuracy_df_list = [] #This list will have the balanced accuracy df's for the 6 problems
    from Lazy_predict_version_Nethali_new1 import Classification
    #%load_ext line_profiler
    label_list = ['bi_goal','bi_feedback', 'bi_reward', 'tir_goal', 'tir_reward' , 'label_7'] 
    number_dict_list = []
    method_dict_list = []
    
    for k in range(1):
        for c in range(6):
            features_for_problem = feature_list[c]
            number_dict_for_problem = {}
            method_dict_for_problem = {}
            label_series = df[label_list[c]] #7/5
            for j in range(15): #(len(features_for_problem)):
                feature_index_list = [x-1 for x in features_for_problem[0:j+1]] #we have to subtract 1 to get the right column feature 1 = column 0 and so on
                balanced_accuracy_for_problem_df_list = [] #this is specific to a given j
                problem_df = df.iloc[:,feature_index_list] #this needs to be changed to pick one feature at a time [0: len(feat)], the new feature matrix 7/5
                
                #This is for a particular j no of features
                
                for i in range(5): #This is for manual cv
                    random_state = [42, 35, 68, 92, 70] 
                    #random_state = [42, 35, 68, 92, 80] 
                   
                    problem_df1 = pd.concat([problem_df, label_series], axis =1) 
                    problem_df2 = problem_df1.dropna() #This is the correct df for the problem with the right subjects
                    X = problem_df2.drop([label_list[c]], axis =1)#get the features with the right samples
                    X= X.astype('float')
                    Y = problem_df2[label_list[c]]
                    Y= Y.astype('int')
                    
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state= random_state[i])
                    clf = Classification(verbose=0,ignore_warnings=True, custom_metric=None)
                    models, predictions = clf.fit( X_train, X_test, Y_train, Y_test)
                    balanced_accuracy_for_problem_df_list.append(models.drop(['Accuracy', 'ROC AUC', 'F1 Score', 'Time Taken', 'Balanced Accuracy','Misclassifications'], axis = 1).rename(columns={"AIC":str(i)}))
                #get the resulting concat df
                result_df = pd.concat(balanced_accuracy_for_problem_df_list, axis=1)
                #get the lowest median 
                number_dict_for_problem[str(j+1)+'features'] = result_df.median(axis= 1).min()
                method_dict_for_problem[str(j+1)+'features'] = result_df.median(axis= 1).idxmin()
            number_dict_list.append(number_dict_for_problem)
            method_dict_list.append(method_dict_for_problem)
    return number_dict_list , method_dict_list 






#get the optimal features for elbow curve
def get_stats_for_elbow_curve2(df, feature_list):
    
    from sklearn.model_selection import train_test_split
    balanced_accuracy_df_list = [] #This list will have the balanced accuracy df's for the 6 problems
    from Lazy_predict_version_Nethali_new import Classification
    #%load_ext line_profiler
    label_list = ['bi_goal','bi_feedback', 'bi_reward', 'tir_goal', 'tir_reward' , 'label_7'] 
    number_dict_list = []
    method_dict_list = []
    
    for c in range(6):
        features_for_problem = feature_list[c]
        number_dict_for_problem = {}
        method_dict_for_problem = {}
        label_series = df[label_list[c]] #7/5
        for j in range(15): #(len(features_for_problem)):
            feature_index_list = [x-1 for x in features_for_problem[0:j+1]] #we have to subtract 1 to get the right column feature 1 = column 0 and so on
            balanced_accuracy_for_problem_df_list = [] #this is specific to a given j
            problem_df = df.iloc[:,feature_index_list] #this needs to be changed to pick one feature at a time [0: len(feat)], the new feature matrix 7/5
                
            #This is for a particular j no of features
            
            for k in range(1):
                                      
                problem_df1 = pd.concat([problem_df, label_series], axis =1) 
                problem_df2 = problem_df1.dropna() #This is the correct df for the problem with the right subjects
                X = problem_df2.drop([label_list[c]], axis =1)#get the features with the right samples
                X= X.astype('float')
                Y = problem_df2[label_list[c]]
                Y= Y.astype('int')
                    
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,  random_state= 42)
                clf = Classification(verbose=0,ignore_warnings=True, custom_metric=None)
                models, predictions = clf.fit( X_train, X_test, Y_train, Y_test)
                balanced_accuracy_for_problem_df_list.append(models.drop(['Accuracy', 'ROC AUC', 'F1 Score', 'Time Taken'], axis = 1).rename(columns={"Balanced Accuracy":str(k)}))
            #get the resulting concat df
            result_df = pd.concat(balanced_accuracy_for_problem_df_list, axis=1)
            #get the heighest median 
            number_dict_for_problem[str(j+1)+'features'] = result_df.median(axis= 1).max()
            method_dict_for_problem[str(j+1)+'features'] = result_df.median(axis= 1).idxmax()
        number_dict_list.append(number_dict_for_problem)
        method_dict_list.append(method_dict_for_problem)
    return number_dict_list , method_dict_list 



       
    
    
#function to get the optimal features from the above number and method list which has results for many random seed iterations

def get_optimal_features_and_methods(number_dict, method_dict):  
#First create a df with the median values for each problem's accuracy
    summary_mean_acc_df = pd.DataFrame()
    for c in range(6):
        problem_accuracies_df = pd.DataFrame()
        for i in range(1): #because we have 4 random seeds in the above function
            problem_accuracies_df[str(i+1)] = number_dict[6*i+c].values() #extracts the correct values from the number_dict
        #Now get the mean of the problem df
        summary_mean_acc_df['C'+ str(c)] = problem_accuracies_df.mean(axis=1)
        
    best_features_methods = []
        
    #Now work on this df to get the max acc and features 
    cols = summary_mean_acc_df.columns
    for k in range(6):
        #get the max of each column and the row index
        max_acc_for_prob = summary_mean_acc_df[cols[k]].max() #max of the problem
        threshold = max_acc_for_prob-0.01
        best_feat_indices = np.where(summary_mean_acc_df[cols[k]] >= threshold)[0].tolist() #row index represents the number of features -1
        max_feats_for_prob = min(best_feat_indices) #max number of features needed is the lowest number of features that is in the threshold
        if max_feats_for_prob == 0:
            best_feat_indices.remove(min(best_feat_indices))
            max_feats_for_prob = min(best_feat_indices)
        max_accu_for_prob = summary_mean_acc_df.iloc[max_feats_for_prob][cols[k]]
        method_for_prob = []
        
        #get the methods corresponding to the features
        for i in range(1):
            dict_problem = method_dict[6*i+k]
            for j in range(len(best_feat_indices)):
                good_method = list(dict_problem.values())[best_feat_indices[j]] #get the value (that is the method) in each index
                method_for_prob.append(good_method)
        best_method = max(method_for_prob, key = method_for_prob.count) #The best method is the one that appeared in majority of the best indices
        #Now record these 3 values
        best_features_methods.append((max_accu_for_prob, max_feats_for_prob+1 ,best_method ))
        
    return best_features_methods

        
        
                
'''


#To get an exclusion criteria, we want to see each time series' completion percentage
# A time series is 100% complete if all of its (imoprtant) values are available for a given trial.
import collections
def compute_completion_per_trial(df):
    completion_percent_list= [] #list of lists. One list consists of completion percentage for its trials. 
    broken_trials = {}
    broken_subject_list = []
    subjects = list(df['subject'].unique())
    for i in range(len(subjects)):
        subject_df = df.loc[df['subject']== subjects[i]].drop(['subject'], axis =1)
        #'sample','time','goal','feedback', 'reward', 'bi_goal', 'bi_reward', 'label_18'], axis =1)
        trials = list(df['trial'].unique())
        subject_completion_list = []
        for j in range(len(trials)):
            trial_df = subject_df.loc[subject_df['trial']== trials[j]]
            data_size = trial_df.shape[0]*(trial_df.shape[1]-1) #do not want to count the trial number column as data
            complete_percentage = (1-(trial_df.isnull().sum().sum()/data_size).round(2)) * 100
            if complete_percentage < 80:
                broken_trials[str(subjects[i])+ 'trial'+ str(trials[j])] = complete_percentage
                broken_subject_list.append(subjects[i])
            subject_completion_list.append(complete_percentage)
        completion_percent_list.append(subject_completion_list)
        #Count the number of times the subject was reported broken
        broken_subjects_sorted = collections.Counter(broken_subject_list).most_common() #counts the occurences and sort them in decreasing order
    return broken_trials, broken_subjects_sorted, completion_percent_list
'''