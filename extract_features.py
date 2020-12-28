from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np


def extract_session_features(session_df):
    user_id = session_df['user_id'].tolist()[0]
    task_id = session_df['task_id'].tolist()[0]
    stage_id = TASK_STAGES[task_id]
    project_id = session_df['project_id'].tolist()[0]
    print(user_id,stage_id,project_id)
    task_num = TASK_STAGETOID[stage_id]

    creation_time_columnname = 'timestamp'


    session_progress_df = pd.read_csv('./stages_progress.csv')
    task_start_time = session_progress_df[(session_progress_df['user_id']==user_id)&(session_progress_df['stage_id']==stage_id)]['created_at'].tolist()[0]
    task_start_time = pd.to_datetime(task_start_time)
    session_df[creation_time_columnname] = pd.to_datetime(session_df[creation_time_columnname])
    session_df['session_time_elapsed']= session_df[creation_time_columnname]-task_start_time
    session_df['session_time_elapsed'] = session_df['session_time_elapsed'].dt.total_seconds()

    contentpages_unique_firstvisit = pd.read_csv('./pages_cleaned.csv',sep='\t')

    contentpages_unique_firstvisit = contentpages_unique_firstvisit[contentpages_unique_firstvisit['project_id']==project_id]
    contentpages_unique_firstvisit = contentpages_unique_firstvisit[contentpages_unique_firstvisit['pages_is_query']==0]
    contentpages_unique_firstvisit = contentpages_unique_firstvisit[contentpages_unique_firstvisit['url']!='https://www.google.com/']
    contentpages_unique_firstvisit = contentpages_unique_firstvisit[~contentpages_unique_firstvisit['url'].str.contains('https://problemhelp.comminfo')]
    contentpages_unique_firstvisit = contentpages_unique_firstvisit[~contentpages_unique_firstvisit['url'].str.contains('http://problemhelp.comminfo')]
    contentpages_unique_firstvisit = contentpages_unique_firstvisit[~contentpages_unique_firstvisit['url'].str.contains('chrome://newtab/')]
    contentpages_unique_firstvisit = contentpages_unique_firstvisit.groupby('url',as_index=False).nth(0)
    contentpages_unique_firstvisit['created_at'] = pd.to_datetime(contentpages_unique_firstvisit['created_at'])
    contentpages_unique_firstvisit = contentpages_unique_firstvisit.sort_values(by='created_at',ascending=True)

    contentpages_unique_firstvisit_set = list(set(contentpages_unique_firstvisit['url'].tolist()))
    session_df['url_test'] = session_df['url']
    session_df['is_content'] = session_df['url'].isin(contentpages_unique_firstvisit_set)

    session_df['is_serp'] = session_df['url'].str.contains('https://www.google.com/search')


    queries_unique_firstvisit = pd.read_csv('./queries_cleaned.csv',sep='\t')
    queries_unique_firstvisit = queries_unique_firstvisit[queries_unique_firstvisit['project_id']==project_id]
    q = []
    for(query,group) in queries_unique_firstvisit.groupby('query'):
        group['query'] = query
        group = group.sort_values(by='created_at',ascending=True)
        q += [group.iloc[0]]
    queries_unique_firstvisit = pd.DataFrame(q)
    queries_unique_firstvisit['created_at'] = pd.to_datetime(queries_unique_firstvisit['created_at'])
    queries_unique_firstvisit = queries_unique_firstvisit.sort_values(by='created_at',ascending=True)
  

    session_df['timestamp_test'] = session_df[creation_time_columnname]

    bookmarks_df = pd.read_csv('./bookmarks_cleaned.csv',sep='\t')
    bookmarks_df = bookmarks_df[bookmarks_df['project_id']==project_id]
    session_df['session_num_bookmarks']= 0

    n_bookmarks = 0
    for (n,row) in bookmarks_df.iterrows():
        bookmark_created_at = pd.to_datetime(row['created_at'])

        n_bookmarks += 1
        session_df.loc[session_df[creation_time_columnname]>=bookmark_created_at,'session_num_bookmarks'] = n_bookmarks

    session_df['session_num_content_distinct'] = 0
    n_pages = 0
    for (n,row) in contentpages_unique_firstvisit.iterrows():
        contentpage_created_at = pd.to_datetime(row['created_at'])
        n_pages += 1
        session_df.loc[session_df[creation_time_columnname]>=contentpage_created_at,'session_num_content_distinct'] = n_pages


    print('session_num_queries_distinct')
    session_df['session_num_queries_distinct'] = 0
    n_queries = 0
    for (n,row) in queries_unique_firstvisit.iterrows():
        querypage_created_at = pd.to_datetime(row['created_at'])
        n_queries += 1
        session_df.loc[session_df[creation_time_columnname]>=querypage_created_at,'session_num_queries_distinct'] = n_queries


    session_df['session_num_content_distinct_perquery']=session_df['session_num_content_distinct']/	session_df['session_num_queries_distinct']
    session_df['session_num_content_distinct_perquery'] = session_df['session_num_content_distinct_perquery'].replace({np.nan:0,np.inf:0})
    return session_df


def extract_actionwise_features(session_df):
    user_id = session_df['user_id'].tolist()[0]
    task_id = session_df['task_id'].tolist()[0]
    stage_id = TASK_STAGES[task_id]
    project_id = session_df['project_id'].tolist()[0]
    print(user_id,stage_id,project_id)
    task_num = TASK_STAGETOID[stage_id]

    creation_time_columnname = 'timestamp'


    session_progress_df = pd.read_csv('./stages_progress.csv')
    task_stop_time = session_progress_df[(session_progress_df['user_id']==user_id)&(session_progress_df['stage_id']==(stage_id+1))]['created_at'].tolist()[0]
    task_stop_time = pd.to_datetime(task_stop_time)

    session_df['time_elapsed_action'] = (pd.to_datetime(session_df[creation_time_columnname].shift(-1))-pd.to_datetime(session_df[creation_time_columnname])).dt.total_seconds()
    
    session_df.iloc[-1,session_df.columns.get_loc('time_elapsed_action')] = (task_stop_time-session_df[creation_time_columnname].tolist()[-1]).total_seconds()

    session_df['session_time_elapsed_afteraction'] = session_df['session_time_elapsed']+session_df['time_elapsed_action']


    session_df['time_elapsed_action_serp'] = session_df['time_elapsed_action']*session_df['is_serp']
    session_df['time_elapsed_action_content'] = session_df['time_elapsed_action']*session_df['is_content']

    session_df['session_time_elapsed_serp'] = session_df['time_elapsed_action_serp'].cumsum()
    session_df['session_time_elapsed_content'] = session_df['time_elapsed_action_content'].cumsum()

    sdf = []
    previous_num_queries_distinct =  None


    for (n,row) in session_df.iterrows():
        if row['actions']=='problem_help':
            row['session_num_queries_distinct'] = previous_num_queries_distinct+1
        else:
            if (previous_num_queries_distinct is not None) and (row['session_num_queries_distinct'] < previous_num_queries_distinct):
                row['session_num_queries_distinct'] = previous_num_queries_distinct

        previous_num_queries_distinct = row['session_num_queries_distinct']
        sdf += [row]

    session_df = pd.DataFrame(sdf)


    prevseg_time_elapsed = None
    prevsession_time_elapsed = None

    prevseg_time_elapsed_serp = None
    prevsession_time_elapsed_serp = None

    prevseg_time_elapsed_content = None
    prevsession_time_elapsed_content = None

    prevseg_num_content_distinct = None
    prevsession_num_content_distinct = None

    prevseg_num_bookmarks = None
    prevsession_num_bookmarks = None

    prevseg_query_len = None
    prevsession_query_len = None

    prevseg_num_query_nocontent = None
    prevsession_num_query_nocontent = None


    session_df['session_query_len_mean'] = 0
    session_df['session_num_query_nocontent'] = 0

    sdf = []
    for(session_num_queries_distinct,group) in session_df.groupby('session_num_queries_distinct'):
        group['session_num_queries_distinct']=session_num_queries_distinct


        queries_segment = group['query'].tolist()
        queries_segment = [q.strip() for q in queries_segment if (type(q)==str) and q!= '0' and q.strip() != '']
        # print(queries_segment)
        query_segment = '' if len(queries_segment)==0 else queries_segment[0]

        session_query_len_mean = group['session_query_len_mean'].tolist()[-1]
        if prevseg_query_len is not None:
            group['seg_query_len']=len(query_segment.split())
            group['prev_query_len']=prevseg_query_len
            group['session_query_len_mean']=(prevsession_query_len*(session_num_queries_distinct-1) + len(query_segment.split()))/session_num_queries_distinct
            prevseg_query_len = len(query_segment.split())
        else:
            group['seg_query_len']=len(query_segment.split())
            group['prev_query_len']=np.nan
            group['session_query_len_mean']=len(query_segment.split())
            prevseg_query_len = len(query_segment.split())
        prevsession_query_len = group['session_query_len_mean'].tolist()[-1]


        session_time_elapsed_serp = group['session_time_elapsed_serp'].tolist()[-1]
        if prevseg_time_elapsed_serp is not None:
            group['seg_time_elapsed_serp']=session_time_elapsed_serp-prevsession_time_elapsed_serp
            group['prev_time_elapsed_serp']=prevseg_time_elapsed_serp
            prevseg_time_elapsed_serp = session_time_elapsed_serp-prevsession_time_elapsed_serp
        else:
            group['seg_time_elapsed_serp']=session_time_elapsed_serp
            group['prev_time_elapsed_serp']=np.nan
            prevseg_time_elapsed_serp = session_time_elapsed_serp
        prevsession_time_elapsed_serp = group['session_time_elapsed_serp'].tolist()[-1]

        session_time_elapsed_content = group['session_time_elapsed_content'].tolist()[-1]
        if prevseg_time_elapsed_content is not None:
            group['seg_time_elapsed_content']=session_time_elapsed_content-prevsession_time_elapsed_content
            group['prev_time_elapsed_content']=prevseg_time_elapsed_content
            prevseg_time_elapsed_content = session_time_elapsed_content-prevsession_time_elapsed_content
        else:
            group['seg_time_elapsed_content']=session_time_elapsed_content
            group['prev_time_elapsed_content']=np.nan
            prevseg_time_elapsed_content = session_time_elapsed_content
        prevsession_time_elapsed_content = group['session_time_elapsed_content'].tolist()[-1]

        session_time_elapsed = group['session_time_elapsed_afteraction'].tolist()[-1]
        if prevseg_time_elapsed is not None:
            group['seg_time_elapsed']=session_time_elapsed-prevsession_time_elapsed
            group['prev_time_elapsed']=prevseg_time_elapsed
            prevseg_time_elapsed = session_time_elapsed-prevsession_time_elapsed
        else:
            group['seg_time_elapsed']=session_time_elapsed
            group['prev_time_elapsed']=np.nan
            prevseg_time_elapsed=session_time_elapsed
        prevsession_time_elapsed = group['session_time_elapsed_afteraction'].tolist()[-1]



        group['session_num_queries_distinct']=session_num_queries_distinct
        session_time_elapsed = group['session_time_elapsed_afteraction'].tolist()[-1]
        if prevseg_time_elapsed is not None:
            group['seg_time_elapsed']=session_time_elapsed-prevsession_time_elapsed
            group['prev_time_elapsed']=prevseg_time_elapsed
            prevseg_time_elapsed = session_time_elapsed-prevsession_time_elapsed
        else:
            group['seg_time_elapsed']=session_time_elapsed
            group['prev_time_elapsed']=np.nan
            prevseg_time_elapsed=session_time_elapsed

        prevsession_time_elapsed = group['session_time_elapsed_afteraction'].tolist()[-1]



        session_num_content_distinct = group['session_num_content_distinct'].tolist()[-1]
        if prevseg_num_content_distinct is not None:
            group['seg_num_content_distinct']=session_num_content_distinct-prevsession_num_content_distinct
            group['prev_num_content_distinct']=prevseg_num_content_distinct
            prevseg_num_content_distinct = session_num_content_distinct-prevsession_num_content_distinct
        else:
            group['seg_num_content_distinct']=session_num_content_distinct
            group['prev_num_content_distinct']=np.nan
            prevseg_num_content_distinct = session_num_content_distinct

        prevsession_num_content_distinct = group['session_num_content_distinct'].tolist()[-1]


        group_nocontent = group['seg_num_content_distinct'].tolist()[-1]==0
        if prevseg_num_query_nocontent is not None:
            group['seg_num_query_noclicks']=int(group_nocontent)
            group['prev_num_query_noclicks']=prevseg_num_query_nocontent
            prevseg_num_query_nocontent = int(group_nocontent)
            prevsession_num_query_nocontent = prevsession_num_query_nocontent+int(group_nocontent)
            group['session_num_query_noclicks'] = prevsession_num_query_nocontent
        else:
            group['seg_num_query_noclicks'] = int(group_nocontent)
            group['prev_num_query_noclicks']=np.nan
            prevsession_num_query_nocontent = int(group_nocontent)
            prevseg_num_query_nocontent = int(group_nocontent)
            group['session_num_query_noclicks'] = prevsession_num_query_nocontent





        session_num_bookmarks = group['session_num_bookmarks'].tolist()[-1]
        if prevseg_num_bookmarks is not None:
            group['seg_num_bookmarks']=session_num_bookmarks-prevsession_num_bookmarks
            group['prev_num_bookmarks']=prevseg_num_bookmarks
            prevseg_num_bookmarks = session_num_bookmarks-prevsession_num_bookmarks
        else:
            group['seg_num_bookmarks']=session_num_bookmarks
            group['prev_num_bookmarks']=np.nan
            prevseg_num_bookmarks = session_num_bookmarks
        prevsession_num_bookmarks = group['session_num_bookmarks'].tolist()[-1]

        sdf += [group]
    session_df = pd.concat(sdf)


    return session_df

def extract_features(session_df):
    session_df = extract_session_features(session_df)
    session_df = extract_actionwise_features(session_df)

    return session_df


TASK_STAGES = [3,15,19]
TASK_STAGETOID = {3:0,15:1,19:2}
TASK_COMPLEXITIES = ['Moderate','High','Low']
TASK_NEEDS = ['Cognitive','Cognitive','Social']

VALID_USERIDS = [97, 123, 37, 70, 113, 127, 110, 112, 49, 116, 126, 118, 119, 59, 125, 94, 101]
if __name__=='__main__':
    cleaned_data = pd.read_csv('./Final.csv')
    cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'])
    cleaned_data = cleaned_data.sort_values(by=['user_id','task_id','timestamp'],ascending=[True,True,True])
    cleaned_data.to_csv('./Final.csv')
    print(cleaned_data.columns.values)
    # print(set(cleaned_data['user_id'].tolist()))
    out_df = []
    for ((user_id,task_id),group) in cleaned_data.groupby(['user_id','task_id']):
        print((user_id,task_id))
        session_data = cleaned_data[(cleaned_data['user_id']==user_id) & (cleaned_data['task_id']==task_id)]
        session_data_withsessionfeatures = extract_features(session_data)
        out_df += [session_data_withsessionfeatures]
    pd.concat(out_df).to_csv('./Final_withfeatures.csv')
