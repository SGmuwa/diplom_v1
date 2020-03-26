#!/usr/bin/env python
# coding: utf-8

# In[334]:


import scipy
import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as sk
pd.options.display.max_rows = 100

class CF:
    def __init__(self,min_sim,min_overlap):
        self.min_sim = min_sim
        self.min_overlap = min_overlap
        
    def build(self, ratings):
        ratings["rating"] = ratings["rating"].astype(float)
        ratings["avg"] = ratings.groupby("user_id")["rating"].transform(lambda x: self.normalize(x))
        
        ratings["user_id"] = ratings["user_id"].astype("category")
        ratings["movie_id"] = ratings["movie_id"].astype("category")
        
        coo = scipy.sparse.coo_matrix((ratings["avg"].astype(float),
                         (ratings["movie_id"].cat.codes.copy(),
                          ratings["user_id"].cat.codes.copy())))
        
        #Удаление сходства с недостаточным перекрытием!
        overlap_matrix = coo.astype(bool).astype(int).dot(coo.transpose().astype(bool).astype(int))
        
        cor = sk.cosine_similarity(coo,dense_output = False)  
        cor = cor.multiply(cor>self.min_sim)
        
        cor = cor.multiply(overlap_matrix > self.min_overlap)
        return cor
    def normalize(self, x):
        x = x.astype(float)
        x_sum = x.sum()
        x_num = x.astype(bool).sum()
        x_mean = x_sum/x_num
        
        if x.std() == 0:
            return 0.0
        return (x-x_mean) / (x.max()-x.min())
ratings_df = pd.read_csv('ml-100k/u.data', names=['user_id', 'movie_id', 'rating', 'timestamp'], sep='\t', encoding='latin-1', header = None)
ratings_df.drop([0], inplace=True)
ratings_df=ratings_df.apply(pd.to_numeric)

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')
cf =  CF(0.2,2)


# In[335]:


import numpy as np
np.set_printoptions(threshold=np.inf)
similarity_table = cf.build(ratings_df).toarray()
# print(similarity_table[100, :])


# In[336]:


test_id = 591#471
user_rated_movies = list(ratings_df[ratings_df["user_id"]==test_id]["movie_id"])
print(user_rated_movies)
movies[movies["movie_id"].isin(user_rated_movies)].iloc[:,[0,1,2,10,11,13,19,21,22,23]]


# In[337]:


ratings_df[ratings_df['user_id'] == test_id]
# ratings_df.loc[(ratings_df['user_id'] == 471) & (ratings_df['movie_id'].isin( user_rated_movies))]["avg"]
# ratings_df.loc[(ratings_df['user_id'] == 471)].iloc[:,[1,4]]


# In[338]:


#Выбираем самые похожие элементы на те, что оценил пользователь
candidate_set = set()
l = []
for i, el in enumerate (user_rated_movies):
    res = similarity_table[el-1,:]
    dict_ = {i+1:res[i] for i in np.nonzero(res)[0] if i+1 not in user_rated_movies}
    dict_ = {k: dict_[k] for k in sorted(dict_, key=dict_.get, reverse=True) }
    dict_res = {}
    for j,key in enumerate(dict_.keys()):
        if(dict_[key]>0.2):
            dict_res[key] = dict_[key]
    l.append(dict_res)
    candidate_set.update(set(dict_res.keys()))
print(l)
print(candidate_set) #это айди элементов похожих


# In[339]:


#Расчёт  оценок подобных элементов (кандидатов)
candidate_items = list(candidate_set)
rec = {}
sim_sum = 0#знаменатель
pre = 0#числитель

rat_u = ratings_df[ratings_df["user_id"]==test_id]
movie_ids = dict(zip(list(rat_u["movie_id"]) ,list(rat_u["rating"]) ))
user_mean = sum(movie_ids.values()) / len(movie_ids)
print(user_mean)

for i, el in enumerate(candidate_items):
    #Для целевого элемента ищем схожесть с другими для дальнейшего прогноза оценки (берем фильмы в окрестности целевого элемента)
    res = similarity_table[el-1,:]
    sim_items = {i+1:res[i] for i in np.nonzero(res)[0] if i+1 != el and i+1 in user_rated_movies}
    pre = 0
    if(len(sim_items)>1):#Желательно иметь более одного оцененного элемента
        mean_ui_df = ratings_df.loc[(ratings_df['user_id'] == test_id) & (ratings_df['movie_id'].isin(sim_items.keys()))]
#         print(el)
#         print(sim_items)
#         print(mean_ui_df)
        
#         print(sim_items.keys())
        sim_sum = sum(sim_items.values())
        if sim_sum>0:
            t=dict(zip(mean_ui_df["movie_id"], mean_ui_df["avg"]))
            pre = sum([sim_items[j] * t[j]  for j in sim_items.keys()])
            rec[el] = {'prediction' : user_mean+pre/sim_sum,
                      'sim_items' : list(sim_items.keys())}
print(rec)


# In[340]:


sorted_items = sorted(rec.items(),key= lambda item: -float(item[1]["prediction"]))[:10]


# In[341]:


sorted_items


# In[343]:


movies[movies["movie_id"]==100]


# In[344]:


print(ratings_df[ratings_df['user_id'] == test_id])


# In[235]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[114]:


from decimal import *
def recommend_item_by_ratings(user_id, active_user_items, num = 6):
    #Словарь из оцененных пользователем фильмов
    movie_ids = dict(zip(list(active_user_items["movie_id"]) ,list(active_user_items["rating"]) ))
    #Расчёт среднего
    user_mean = Decimal(sum(movie_ids.values())) / Decimal(len(movie_ids))
    print(user_mean)
    


# In[133]:


test_id = 471
active_user_items = ratings_df[ratings_df["user_id"]==test_id]
print(active_user_items)


# In[116]:


recommend_item_by_ratings(user_id = 470, active_user_items = active_user_items)

