import pandas as pd


# pd.set_option('display.max_columns', None)
# data = {'product_id': ['B01MTEI8M6', 'B0853X2F4M', 'B07R91S92W', 'B0853WVZ6Q', 'B07R51S57D'],
#         'product_type': ['SHOES', 'CELLULAR_PHONE_CASE', 'CELLULAR_PHONE_CASE', 'CELLULAR_PHONE_CASE', 'FURNITURE_COVER'],
#         'image_ids': [('71C4hQAAs2L', '718uEco1DAL', '71BMHcaG5GL', '7105JsM1HYL', '71F5emAT2DL', '81KXLckXuxL', '71iDTnkSlsL', '714CmIfKIYL'),
#                       ('51wiRu6gT9L', '51DEYNWtfsL', '81RoDPeqygL', '518RypAtk8L', '81+4dBN1jsL'),
#                       ('618uWaH5elL', '61hUyEcVYtL', '61ajxcF6Y1L', '613aK9gtKuL', '614jfuPGINL', '61LWeNhjZ9L'),
#                       ('71QdPoaGvYL', '51wiRu6gT9L', '51DEYNWtfsL', '518RypAtk8L', '71tgJqobw6L'),
#                       ('81hrbJmpSDL', '71vIxbNBCnL', '81J1xbXlidL', '81LRCuVt2uL', '81ZpWQjjBbL', '81D1kbLhn-L', '81oqJP-fuYL', '71E62qS--6L')]}
#
#
# ids = [i for j in data['image_ids'] for i in j]
# tot = len(ids)
# unique = len(set(ids))
# reps = tot - unique
# print(f'ci sono in totale {tot} image_id')
# print(f'di questi, {unique} sono unici')
# print(f'nel dataframe devono rimanere {tot - 2*reps}')
#
# df = pd.DataFrame(data)
# print(df)
# print('-'*100)
#
# df = df.explode('image_ids')
# df.drop_duplicates(subset='image_ids', inplace=True, ignore_index=True, keep=False)
# df.set_index('image_ids', inplace=True)
#
# print(df)
# print('-'*100)
#
# df = df.groupby(['product_id', 'product_type']).agg({'image_ids': lambda x: x.to_list()}).reset_index()
# print(df)
# print('-'*100)
#
# df = df[df['image_ids'].str.len() >= 5].reset_index()
# print(df)
#
# data1 = {
#     'product_id': [1, 2, 3, 4],
#     'product_type': ['Type A', 'Type B', 'Type C', 'Type D'],
#     'image_ids': [('image1', 'image2'), ('image5'), ('image6'), ('image7')]
# }
# df1 = pd.DataFrame(data1).explode('image_ids').set_index('image_ids')
# print(df1)
# print('-'*100)
#
# data2 = {
#     'height': [10, 15, 12, 8],
#     'width': [5, 8, 6, 9],
#     'path': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'],
#     'image_ids': [('image1', 'image2'), ('image3', 'image4', 'image5'), ('image6'), ('image7', 'image8')]
# }
# df2 = pd.DataFrame(data2).explode('image_ids').set_index('image_ids')
# print(df2)
# print('-'*100)
#
# merged_df = df1.merge(df2, how='left', on='image_ids')
# print(merged_df)
# print('-'*100)
#
# top_n = merged_df['product_type'].value_counts()[0:2]
# top_n = top_n.index.to_list()
# print(top_n)
#
# import pandas as pd
#
#
# a = pd.DataFrame({'col': ['add', 'fgs', 'ser']}, index=['fgr', 'sfg', 'dvd'])
# print(type(a.loc['sfg', 'col']))
#
# j = 10
# files = [1,2,3,4]
# print([j] * len(files))
#
# import torch
#
#
# t = torch.tensor([[1.0, 2.1, 1.5, 2.6]])
# o = torch.nn.functional.softmax(t, dim=1)
# a = torch.argmax(o, dim=1)
# print(o, a)
#
# a = [1, 2, 3, 4, 5]
#
# print(a[0:0])

import os
from pathlib import Path


path = Path(r'C:\Users\rx571gt-b034t\Desktop\PROJECT\code\runs')
print(os.listdir(path))
