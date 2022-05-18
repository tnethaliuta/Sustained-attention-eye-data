#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#population regression pipeline

#Load the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

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
        columns = subject_df.drop(['subject', 'rt'], axis =1).columns #the columns in the above df that needs to be standardized 
        for j in range(len(columns)): #standardize each column
            subject_df[columns[j]] = scaler.fit_transform(subject_df[columns[j]].values.reshape(-1,1))
        list_of_subject_dfs.append(subject_df)
    new_df = pd.concat(list_of_subject_dfs)
    return new_df


#Function for feature expansion using polynomial features
def feature_expansion(X):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    new_x = poly.fit_transform(X)
    X_1= pd.DataFrame(data = new_x, columns = poly.get_feature_names(X.columns)) #get_feature_names is a method of poly
    return X_1




#Add the labels back to the feature matrix

def get_feature_matrix_with_labels(df, df1):
    import itertools
    subjects = df1['subject'].unique()
    
    goal_list = []
    feedback_list = []
    reward_list = []
    label_names = ['goal', 'feedback', 'reward']
    label_list = [goal_list, feedback_list, reward_list]

    for i in range (len(subjects)):
        subject_size = df1.loc[df1['subject'] == subjects[i]].shape[0]
        subject_df = df.loc[df['subject'] == subjects[i]].head(1)
        for j in range(3):
            label_list[j].append([subject_df.iloc[0][label_names[j]]]*subject_size)

    for i in range(3):
        df1[label_names[i]] = list(itertools.chain(*label_list[i]))

    return df1


import pandas as pd
import statsmodels.api as sm


def forward_regression(X, y,
                       threshold_in,
                       verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_regression(X, y,
                           threshold_out,
                           verbose=False):
    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


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