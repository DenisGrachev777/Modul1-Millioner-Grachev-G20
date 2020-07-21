#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# In[364]:


df = pd.read_csv('movie_bd_v5.csv')
df.sample(5)


# In[365]:


df.describe()


# # Предобработка

# In[471]:


answers = {'Ответ №':[5,2,3,2,1,5,5,1,4,5,3,1,5,3,3,3,2,1,5,1,4,2,5,5,3,1,5],'Полный ответ':['Pirates of the Caribbean: On Stranger Tides (tt1298650)','Gods and Generals (tt0279111)','Winnie the Pooh (tt1449283)',110,107,'Avatar (tt0499549)','The Lone Ranger (tt1210819)',1478,'The Dark Knight (tt0468569)','The Lone Ranger (tt1210819)','Drama','Drama','Peter Jackson','Robert Rodriguez','Chris Hemsworth','Matt Damon','Action','K-19: The Widowmaker (tt0267626)',2015,2014,'Сентябрь',450,'Peter Jackson','Four By Two Productions','Midnight Picture Show','Inside Out, The Dark Knight, 12 Years a Slave','Daniel Radcliffe & Rupert Grint '],'Верный ли ответ':['+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+']} 
answers = pd.DataFrame(answers, index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
# создадим словарь для ответов

# тут другие ваши предобработки колонок например:

#the time given in the dataset is in string format.
#So we need to change this in datetime format
# ...


# In[472]:


answers


# In[234]:


df.info()


# In[366]:


df.isna().sum()


# In[367]:


dfc = df.copy(deep=True)
dfc.head(1)


# In[237]:


df['profit']= df.revenue - df.budget
df.head(3)


# # 1. У какого фильма из списка самый большой бюджет?

# Использовать варианты ответов в коде решения запрещено.    
# Вы думаете и в жизни у вас будут варианты ответов?)

# In[238]:


# в словарь вставляем номер вопроса и ваш ответ на него
# Пример: 
answers['1'] = '2. Spider-Man 3 (tt0413300)'
# запишите свой вариант ответа
answers['1'] = '...'
# если ответили верно, можете добавить комментарий со значком "+"


# In[239]:


# тут пишем ваш код для решения данного вопроса:
df.loc[df['budget'].idxmax()]['original_title']


# ВАРИАНТ 2

# In[240]:


# можно добавлять разные варианты решения


# # 2. Какой из фильмов самый длительный (в минутах)?

# In[241]:


# думаю логику работы с этим словарем вы уже поняли, 
# по этому не буду больше его дублировать
answers['2'] = '...'


# In[242]:


df.loc[df['runtime'].idxmax()]['original_title']


# # 3. Какой из фильмов самый короткий (в минутах)?
# 
# 
# 
# 

# In[243]:


df.loc[df['runtime'].idxmin()]['original_title']


# # 4. Какова средняя длительность фильмов?
# 

# In[244]:


df.agg('mean')['runtime'].round(0)


# # 5. Каково медианное значение длительности фильмов? 

# In[245]:


df.agg('median')['runtime']


# # 6. Какой самый прибыльный фильм?
# #### Внимание! Здесь и далее под «прибылью» или «убытками» понимается разность между сборами и бюджетом фильма. (прибыль = сборы - бюджет) в нашем датасете это будет (profit = revenue - budget) 

# In[246]:


# лучше код получения столбца profit вынести в Предобработку что в начале
df.loc[df['profit'].idxmax()]['original_title']


# # 7. Какой фильм самый убыточный? 

# In[247]:


df.loc[df['profit'].idxmin()]['original_title']


# # 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?

# In[248]:


df1 = df.loc[df.revenue> df.budget]
len(df1)


# # 9. Какой фильм оказался самым кассовым в 2008 году?

# In[249]:


df2 = df[df.release_year == 2008].sort_values(['profit'],ascending=False).head(1)
df2


# # 10. Самый убыточный фильм за период с 2012 по 2014 г. (включительно)?
# 

# In[250]:


df2 = df[(df.release_year >2011)&(df.release_year <2015)].sort_values(['profit'], ascending=True).head(1)
df2


# # 11. Какого жанра фильмов больше всего?

# In[251]:


# эту задачу тоже можно решать разными подходами, попробуй реализовать разные варианты
# если будешь добавлять функцию - выноси ее в предобработку что в начале

new_df = df['genres'].str.split('|',expand=True).stack().value_counts()
new_df.head(3)


# ВАРИАНТ 2

# In[252]:


more_genres = df[['genres']]
more_genres.genres = more_genres.genres.apply(lambda s:s.split('|'))
more_genres_explode = more_genres.explode('genres')
more_genres_explode.stack().value_counts().head(2)


# # 12. Фильмы какого жанра чаще всего становятся прибыльными? 

# In[253]:


data3 = df[df.profit >0]
new_df3 = data3['genres'].str.split('|',expand=True).stack().value_counts()
new_df3.head(3)


# # 13. У какого режиссера самые большие суммарные кассовые сбооры?

# In[254]:


grouped_df = df.groupby(['director'])['revenue'].sum().sort_values(ascending=False)
print(grouped_df.head(3))


# # 14. Какой режисер снял больше всего фильмов в стиле Action?

# In[255]:


grouped1 = df[df.genres.str.contains("Action", na=False)]
grouped2 = grouped1['director'].str.split('|',expand=True).stack().value_counts()
grouped2


# # 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году? 

# In[256]:


data_sub = df[df.release_year == 2012][['cast', 'revenue']]
data_sub.cast = data_sub.cast.apply(lambda s: s.split('|'))
data_sub_exploded = data_sub.explode('cast')
data_sub_exploded.groupby(by = 'cast').revenue.sum().sort_values(ascending = False)


# # 16. Какой актер снялся в большем количестве высокобюджетных фильмов?

# In[257]:


df161 = df
def counter(movie_bd, x):
    data_plot = movie_bd[x].str.cat(sep='|')
    dat = pd.Series(data_plot.split('|'))
    info = dat.value_counts(ascending=False)
    return info
sum_gen = counter(df161[df161['budget'] > df161['budget'].mean()], 'cast')
sum_gen
    


# # 17. В фильмах какого жанра больше всего снимался Nicolas Cage? 

# In[258]:


df17 = df[df.cast.str.contains('Nicolas Cage', na=False)]
df17.cast = df17['cast'].str.split('|',expand=True)
df17.genres = df17['genres'].str.split('|',expand=True)
df17.genres.value_counts()


# # 18. Самый убыточный фильм от Paramount Pictures

# In[259]:


df18 = df[df.production_companies.str.contains('Paramount Pictures', na=False)]
df18.production_companies = df17['production_companies'].str.split('|',expand=True)
df18.loc[df18['profit'].idxmin()]['original_title']


# # 19. Какой год стал самым успешным по суммарным кассовым сборам?

# In[284]:


df19 = df[['release_year', 'revenue']]
df19.groupby(by = 'release_year').revenue.sum().sort_values(ascending = False)


# # 20. Какой самый прибыльный год для студии Warner Bros?

# In[342]:


df20 = df[df.production_companies.str.contains('Warner', na=False)]
df201 = df20[['release_year', 'profit']]
df201.groupby(by = 'release_year').profit.sum().sort_values(ascending = False)


# # 21. В каком месяце за все годы суммарно вышло больше всего фильмов?

# In[385]:


df212 = df[['release_date', 'original_title']]
df212['release_date'] = pd.to_datetime(df212['release_date'])
df212['release_date'] = df212.release_date
df212.release_date.dt.month.value_counts().head(2)


# # 22. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)

# In[396]:


df22 = df212
df221 = df22[df22.release_date.dt.month.isin([6, 7, 8,])]
df221.info()


# # 23. Для какого режиссера зима – самое продуктивное время года? 

# In[435]:


df['release_date'] = df22['release_date']
df231 = df[['director', 'release_date']]
df2312 = df231
df23121 = df2312[df2312.release_date.dt.month.isin([12, 1, 2,])]
df23121.director.value_counts().head(2)


# # 24. Какая студия дает самые длинные названия своим фильмам по количеству символов?

# In[440]:


df['lenght'] = df.original_title.map(lambda x: len(x))
df24 = pd.DataFrame(df.production_companies.str.split('|').tolist()).stack()
df24 = df24.value_counts(ascending=False)
df24 = pd.DataFrame(df24)
df24.columns = ['lenght']
for i in df24.index:
    df24.lenght[i] = df[df['production_companies'].map(lambda x: True if i in x else False)].lenght.mean()
df24.sort_values(ascending=False, by = 'lenght')


# In[447]:


#вариант 2
df241 = df
df241['totalwords'] = df['original_title'].str.split().str.len()
df241.sort_values(ascending=False, by = 'totalwords').head(1)
df241.iloc[1448][10].split('|')


# # 25. Описание фильмов какой студии в среднем самые длинные по количеству слов?

# In[458]:


df251 = df
df251['totalwordsover'] = df['overview'].str.split().str.len()
df251.sort_values(ascending=False, by = 'totalwordsover')
df251.iloc[711][10].split('|')


# # 26. Какие фильмы входят в 1 процент лучших по рейтингу? 
# по vote_average

# In[460]:


df26 = df
df26.quantile(0.99, numeric_only=True)['vote_average']
df26[['original_title','vote_average']].sort_values('vote_average',ascending = False).head(int(len(df26['vote_average'])*0.01))


# # 27. Какие актеры чаще всего снимаются в одном фильме вместе?
# 

# In[466]:


from itertools import combinations
actor_list = df.cast.str.split('|').tolist()
combo_list=[]
for i in actor_list:
    for j in combinations(i, 2):
        combo_list.append(' '.join(j))
combo_list = pd.DataFrame(combo_list)
combo_list.columns = ['actor_combinations']
combo_list.actor_combinations.value_counts().head(10)


# # Submission

# In[473]:


# в конце можно посмотреть свои ответы к каждому вопросу
answers


# In[ ]:


# и убедиться что ни чего не пропустил)
len(answers)


# In[ ]:





# In[ ]:




