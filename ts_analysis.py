import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
from time import time
warnings.filterwarnings("ignore")


df = pd.read_csv('covid_state_9_27.csv')

df = df.groupby(['state']).max()
print(df)

t1=time()
'''STEP 1: Pre-processing the dataframes'''
# NY live COVID data
url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
df = pd.read_csv(url, parse_dates=[0])  # Cols: date, state, fips, cases, deaths 

# State population
url = 'https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population'
pop_df = pd.read_html(url)[0]
pop_df = pop_df.iloc[:,2:4]
pop_df = pop_df.droplevel(0, axis=1)
pop_df = pop_df[:52]

def pre_process(total_pop):
    '''Pre-process population values where in some cases the references might 
    have '...[..]' pattern signifying reference
    '''
    try:
        num = int(total_pop)
    except:
        m = re.match(r'\d+[\d+]', total_pop)
        num = int(m.group(0))

    return num

pop_df.rename(columns={pop_df.columns[1]: 'Recent_population'}, inplace=True)
pop_df['Recent_population'] = pop_df['Recent_population'].apply(pre_process)
population_dict = pop_df.set_index('State').T.to_dict('list')

remove_states = []
for state in df.state.unique():
    if state not in population_dict.keys():
        remove_states.append(state)
        
df=df[~df.state.isin(remove_states)]
# print(df)
t2=time()

# print(t2-t1)
'''STEP 2: Merging the dataframes'''
