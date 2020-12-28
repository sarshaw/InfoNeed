from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import scipy.stats.mstats as mstats
import scipy.stats as stats


PREV_FEATURES = ['prev_time_elapsed','prev_num_content_distinct','prev_num_bookmarks',
    'prev_time_elapsed_serp','prev_time_elapsed_content','prev_query_len','prev_num_query_noclicks']
CURRENT_FEATURES = ['reformulation_type_Generalization','reformulation_type_Specification','reformulation_type_WordSubstitution','reformulation_type_Repeat',
                'reformulation_type_New','reformulation_type_SpellingCorrection','reformulation_type_StemIdentical',
                'seg_time_elapsed','seg_num_content_distinct','seg_num_bookmarks','seg_time_elapsed_serp',
                'seg_time_elapsed_content','seg_query_len','seg_num_query_noclicks']
SESSION_FEATURES = ['session_time_elapsed','session_num_bookmarks','session_num_content_distinct',
    'session_num_queries_distinct','session_time_elapsed_serp','session_time_elapsed_content',
    'session_num_content_distinct_perquery','session_query_len_mean','session_num_query_noclicks']
PROBLEMS = ['probhelp_difficult_articulate','probhelp_irrelevant_results','probhelp_topknowledge_lack',
    'probhelp_patience_lack','probhelp_credibility_uncertain','probhelp_sources_unaware',
    'probhelp_toomuch_information','probhelp_source_unavailable','probhelp_no_problem']
# ALL_FEATURES = PREV_FEATURES+CURRENT_FEATURES+SESSION_FEATURES+PROBLEMS
ALL_FEATURES = PREV_FEATURES+SESSION_FEATURES+PROBLEMS
HELPS = ['probhelp_page_recommendation','probhelp_people_recommendation',
    'probhelp_query_recommendation','probhelp_strategy_recommendation','probhelp_no_help_needed','probhelp_system_unsatisfactory']

N_TESTS = 100

full_df = pd.read_csv('./all_problemhelp_features.csv')
prevonly_df = pd.read_csv('./prevanalysis_problemhelp_features.csv')
#
# print(full_df.columns.values)
# for (n,group) in full_df.groupby('task_id'):
#     print("TASK",n)
#     print(len(group.index))
#
#
#
# for (n,group) in full_df.groupby('task_id'):
#     print("TASK",n)
#     print(len(group.index))
# exit()

full_df['reformulation_type_Generalization'] = (full_df['reformulation_type']==1).astype(int)
prevonly_df['reformulation_type_Generalization'] = (prevonly_df['reformulation_type']==1).astype(int)

full_df['reformulation_type_Specification'] = (full_df['reformulation_type']==2).astype(int)
prevonly_df['reformulation_type_Specification'] = (prevonly_df['reformulation_type']==2).astype(int)

full_df['reformulation_type_WordSubstitution'] = (full_df['reformulation_type']==3).astype(int)
prevonly_df['reformulation_type_WordSubstitution'] = (prevonly_df['reformulation_type']==3).astype(int)

full_df['reformulation_type_Repeat'] = (full_df['reformulation_type']==4).astype(int)
prevonly_df['reformulation_type_Repeat'] = (prevonly_df['reformulation_type']==4).astype(int)

full_df['reformulation_type_New'] = (full_df['reformulation_type']==5).astype(int)
prevonly_df['reformulation_type_New'] = (prevonly_df['reformulation_type']==5).astype(int)

full_df['reformulation_type_SpellingCorrection'] = (full_df['reformulation_type']==6).astype(int)
prevonly_df['reformulation_type_SpellingCorrection'] = (prevonly_df['reformulation_type']==6).astype(int)

full_df['reformulation_type_StemIdentical'] = (full_df['reformulation_type']==7).astype(int)
prevonly_df['reformulation_type_StemIdentical'] = (prevonly_df['reformulation_type']==7).astype(int)



out_scores = dict()

# d = pd.DataFrame(prevonly_df,
#                  columns=HELPS+PROBLEMS)
# # Compute the correlation matrix
# corr = d.corr()
# corr.to_csv('test.csv')
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr)
# plt.show()
# exit()




out_df = []
for (featurename,features,df) in [('prev',PREV_FEATURES,prevonly_df),#('curr',CURRENT_FEATURES,prevonly_df),
    ('session',SESSION_FEATURES,prevonly_df),('problems',PROBLEMS,prevonly_df),('all',ALL_FEATURES,prevonly_df)]:
    print(featurename)
    for h in HELPS:
        print(h)
        accuracies = []
        precisions = []
        recalls = []
        fs = []
        for n in range(N_TESTS):
            label_encoder = preprocessing.LabelEncoder()
            X = df[features]
            y = df[h]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = X_train.as_matrix()
            X_test = X_test.as_matrix()

            scaler = StandardScaler()
            y = y.as_matrix()
            label_encoder.fit(y)
            y_train = y_train.tolist()
            y_train = label_encoder.transform(y_train)
            y_test = y_test.tolist()
            y_test = label_encoder.transform(y_test)

            anovakbest_filter = SelectKBest(f_classif, k=min([10, len(features)]))

            model = Pipeline([
                ('scaler',scaler),
                ('anova', anovakbest_filter),
                ('clf', LogisticRegression())
            ])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies += [metrics.accuracy_score(y_test, y_pred)]
            precisions += [metrics.precision_score(y_test, y_pred)]
            recalls += [metrics.recall_score(y_test, y_pred)]
            fs += [metrics.f1_score(y_test,y_pred)]

        out_df += [{'features':featurename,'help':h,'accuracy':np.mean(accuracies),'precision':np.mean(precisions),'recall':np.mean(recalls),'f1':np.mean(fs)}]
        out_scores[('lr',featurename,h,'accuracy')] = accuracies
        out_scores[('lr',featurename,h,'precision')] = precisions
        out_scores[('lr',featurename,h,'recall')] = recalls
        out_scores[('lr',featurename,h,'f1')] = fs

# print(pd.DataFrame(out_df))
pd.DataFrame(out_df).pivot(index='features', columns='help', values='accuracy').round(decimals=2).to_csv('dct_accuracy.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='precision').round(decimals=2).to_csv('dct_precision.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='recall').round(decimals=2).to_csv('dct_recall.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='f1').round(decimals=2).to_csv('dct_f1.csv')

# exit()

out_df = []
for (featurename,features,df) in [('prev',PREV_FEATURES,prevonly_df),#('curr',CURRENT_FEATURES,prevonly_df),
    ('session',SESSION_FEATURES,prevonly_df),('problems',PROBLEMS,prevonly_df),('all',ALL_FEATURES,prevonly_df)]:
    print(featurename)
    for h in HELPS:
        print(h)
        accuracies = []
        precisions = []
        recalls = []
        fs = []
        for n in range(N_TESTS):
            label_encoder = preprocessing.LabelEncoder()
            X = df[features]
            y = df[h]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = X_train.as_matrix()
            X_test = X_test.as_matrix()

            # scaler = StandardScaler()
            y = y.as_matrix()
            label_encoder.fit(y)
            y_train = y_train.tolist()
            y_train = label_encoder.transform(y_train)
            y_test = y_test.tolist()
            y_test = label_encoder.transform(y_test)

            model = DummyClassifier(strategy='most_frequent', random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # print("ytest",y_test)
            # print("y_pred",y_pred)
            accuracies += [metrics.accuracy_score(y_test, y_pred)]
            precisions += [metrics.precision_score(y_test, y_pred)]
            recalls += [metrics.recall_score(y_test, y_pred)]
            fs += [metrics.f1_score(y_test,y_pred)]

        # print(accuracies)
        # print(precisions)
        # print(recalls)
        out_df += [{'features':featurename,'help':h,'accuracy':np.mean(accuracies),'precision':np.mean(precisions),'recall':np.mean(recalls),'f1':np.mean(fs)}]
        out_scores[('mfq',featurename,h,'accuracy')] = accuracies
        out_scores[('mfq',featurename,h,'precision')] = precisions
        out_scores[('mfq',featurename,h,'recall')] = recalls
        out_scores[('mfq',featurename,h,'f1')] = fs

pd.DataFrame(out_df).pivot(index='features', columns='help', values='accuracy').round(decimals=2).to_csv('mfq_accuracy.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='precision').round(decimals=2).to_csv('mfq_precision.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='recall').round(decimals=2).to_csv('mfq_recall.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='f1').round(decimals=2).to_csv('mfq_f1.csv')



out_df = []
for (featurename,features,df) in [('prev',PREV_FEATURES,prevonly_df),#('curr',CURRENT_FEATURES,prevonly_df),
    ('session',SESSION_FEATURES,prevonly_df),('problems',PROBLEMS,prevonly_df),('all',ALL_FEATURES,prevonly_df)]:
    print(featurename)
    for h in HELPS:
        print(h)
        accuracies = []
        precisions = []
        recalls = []
        fs = []
        for n in range(N_TESTS):
            label_encoder = preprocessing.LabelEncoder()
            X = df[features]
            y = df[h]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = X_train.as_matrix()
            X_test = X_test.as_matrix()

            # scaler = StandardScaler()
            y = y.as_matrix()
            label_encoder.fit(y)
            y_train = y_train.tolist()
            y_train = label_encoder.transform(y_train)
            y_test = y_test.tolist()
            y_test = label_encoder.transform(y_test)

            model = DummyClassifier(strategy='stratified',random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # print("ytest",y_test)
            # print("y_pred",y_pred)
            accuracies += [metrics.accuracy_score(y_test, y_pred)]
            precisions += [metrics.precision_score(y_test, y_pred)]
            recalls += [metrics.recall_score(y_test, y_pred)]
            fs += [metrics.f1_score(y_test,y_pred)]

        # print(accuracies)
        # print(precisions)
        # print(recalls)
        out_df += [{'features':featurename,'help':h,'accuracy':np.mean(accuracies),'precision':np.mean(precisions),'recall':np.mean(recalls),'f1':np.mean(fs)}]
        out_scores[('rand',featurename,h,'accuracy')] = accuracies
        out_scores[('rand',featurename,h,'precision')] = precisions
        out_scores[('rand',featurename,h,'recall')] = recalls
        out_scores[('rand',featurename,h,'f1')] = fs

pd.DataFrame(out_df).pivot(index='features', columns='help', values='accuracy').round(decimals=2).to_csv('rand_accuracy.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='precision').round(decimals=2).to_csv('rand_precision.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='recall').round(decimals=2).to_csv('rand_recall.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='f1').round(decimals=2).to_csv('rand_f1.csv')



out_df = []
for (featurename,features,df) in [('prev',PREV_FEATURES,prevonly_df),#('curr',CURRENT_FEATURES,prevonly_df),
    ('session',SESSION_FEATURES,prevonly_df),('problems',PROBLEMS,prevonly_df),('all',ALL_FEATURES,prevonly_df)]:
    print(featurename)
    for h in HELPS:
        print(h)
        accuracies = []
        precisions = []
        recalls = []
        fs = []
        for n in range(N_TESTS):
            label_encoder = preprocessing.LabelEncoder()
            X = df[features]
            y = df[h]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = X_train.as_matrix()
            X_test = X_test.as_matrix()

            # scaler = StandardScaler()
            y = y.as_matrix()
            label_encoder.fit(y)
            y_train = y_train.tolist()
            y_train = label_encoder.transform(y_train)
            y_test = y_test.tolist()
            y_test = label_encoder.transform(y_test)

            model = DummyClassifier(strategy='constant',random_state=42,constant=1)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # print("ytest",y_test)
            # print("y_pred",y_pred)
            accuracies += [metrics.accuracy_score(y_test, y_pred)]
            precisions += [metrics.precision_score(y_test, y_pred)]
            recalls += [metrics.recall_score(y_test, y_pred)]
            fs += [metrics.f1_score(y_test,y_pred)]

        # print(accuracies)
        # print(precisions)
        # print(recalls)
        out_df += [{'features':featurename,'help':h,'accuracy':np.mean(accuracies),'precision':np.mean(precisions),'recall':np.mean(recalls),'f1':np.mean(fs)}]
        out_scores[('const',featurename,h,'accuracy')] = accuracies
        out_scores[('const',featurename,h,'precision')] = precisions
        out_scores[('const',featurename,h,'recall')] = recalls
        out_scores[('const',featurename,h,'f1')] = fs

pd.DataFrame(out_df).pivot(index='features', columns='help', values='accuracy').round(decimals=2).to_csv('const_accuracy.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='precision').round(decimals=2).to_csv('const_precision.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='recall').round(decimals=2).to_csv('const_recall.csv')
pd.DataFrame(out_df).pivot(index='features', columns='help', values='f1').round(decimals=2).to_csv('const_f1.csv')

# HELPS = ['probhelp_page_recommendation','probhelp_people_recommendation',
    # 'probhelp_query_recommendation','probhelp_strategy_recommendation','probhelp_no_help_needed','probhelp_system_unsatisfactory']
print(out_scores.keys())
for (algo1,algo2,feature,help,score) in [
    # ('lr','mfq','problems','probhelp_no_help_needed','accuracy'),
    # ('lr','mfq','all','probhelp_no_help_needed','accuracy'),
    # ('mfq','lr','session','probhelp_page_recommendation','accuracy'),
    # ('lr','mfq','prev','probhelp_people_recommendation','accuracy'),
    # ('lr','mfq','session','probhelp_people_recommendation','accuracy'),
    # ('lr','mfq','all','probhelp_query_recommendation','accuracy'),
    # ('mfq','lr','session','probhelp_strategy_recommendation','accuracy'),
    # ('lr','mfq','session','probhelp_system_unsatisfactory','accuracy'),



    # ('lr','const','problems','probhelp_no_help_needed','precision'),
    # ('lr','const','all','probhelp_no_help_needed','precision'),
    # ('lr','const','session','probhelp_page_recommendation','precision'),
    # ('lr','const','prev','probhelp_people_recommendation','precision'),
    # ('lr','const','session','probhelp_people_recommendation','precision'),
    # ('lr','const','all','probhelp_query_recommendation','precision'),
    # ('lr','rand','session','probhelp_strategy_recommendation','precision'),
    # ('lr','rand','session','probhelp_system_unsatisfactory','precision'),
    #
    # ('const','lr','problems','probhelp_no_help_needed','recall'),
    # ('const','lr','session','probhelp_page_recommendation','recall'),
    # ('const','lr','all','probhelp_people_recommendation','recall'),
    # ('const','lr','all','probhelp_query_recommendation','recall'),
    # ('const','lr','session','probhelp_strategy_recommendation','recall'),
    # ('const','lr','all','probhelp_system_unsatisfactory','recall'),

    ('lr','const','problems','probhelp_no_help_needed','f1'),
    ('lr','const','all','probhelp_no_help_needed','f1'),
    ('lr','const','session','probhelp_page_recommendation','f1'),
    ('const','lr','prev','probhelp_people_recommendation','f1'),
    ('lr','const','all','probhelp_query_recommendation','f1'),
    ('lr','const','session','probhelp_strategy_recommendation','f1'),
    ('lr','rand','session','probhelp_system_unsatisfactory','f1'),

    ]:
    scores1 = out_scores[(algo1,feature,help,score)]
    scores2 = out_scores[(algo2,feature,help,score)]
    print(algo1,algo2,feature,help)
    (stat, p) = stats.ttest_ind(scores1, scores2)
    print('\t',np.mean(scores1),np.mean(scores2),p)
