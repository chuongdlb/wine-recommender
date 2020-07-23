
import datetime as dt
import pandas as pd
import numpy as np
# from numpy.random import randn
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
# from pandas_datareader import data, wb
from faker.providers.person.en import Provider
import random



def preprocess(df, user_pref = 'Both'):
    df = df[df['On-PremiseExclusive'] != 'Y']
    df = df[df['Size(ml)'] == 750]
    df.dropna(subset=['Price'], inplace=True)
    if 'White' == user_pref:
        return df[df['Type'] == 'White'].fillna(0)
    elif 'Red' == user_pref:
        return df[df['Type'] == 'Red'].fillna(0)
    else: 
        return df.fillna(0)

def normalize_vivino_score(df):
    df['VivinoRating'] = df['VivinoRating'] / df['Price']
    vivino_max = df['VivinoRating'].max()
    vivino_min = df['VivinoRating'].min()
    df['VivinoRating'] = (df['VivinoRating'] - vivino_min) / (vivino_max - vivino_min)
    return df

def recommend_using_cosine_similarity(df, user_wine_features):
    df['Similarity'] = cosine_similarity(df[['Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic']], user_wine_features)
    df['BlendedScore'] = df['Similarity'] * .5 + df['VivinoRating'] * .5
    return df.sort_values(by='BlendedScore', ascending=False)

def compute_recommend_for_user(product_data_frame, user_data_frame):
    
    df = pd.DataFrame(columns=['Username', 'WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity', 'WeightRange'])
    for index, row in user_data_frame.iterrows():
        # pre-process
        p_df = preprocess(product_data_frame, row['Type'])
        # normalize Vivino score
        p_df = normalize_vivino_score(p_df)

        out_df = recommend_using_cosine_similarity(p_df, np.array([[row['Dry-Sweet'], row['Light-Bold'],row['Soft-Acidic'],row['Smooth-Tannic']]]))
       
        top_weight_0_9 = out_df[out_df['Similarity'] >= 0.9][['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head()
        top_weight_0_9['Username'] = np.full((top_weight_0_9['WineName'].count(), 1), row['Username'])
        top_weight_0_9['WeightRange'] = np.full((top_weight_0_9['WineName'].count(), 1), ">=0.9")

        for i, r in top_weight_0_9.iterrows():
            df = df.append(r.to_dict(), ignore_index=True)

        top_weight_8_9 = out_df.query('Similarity >= 0.8 & Similarity < 0.9')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head()
        top_weight_8_9['Username'] = np.full((top_weight_8_9['WineName'].count(), 1), row['Username'])
        top_weight_8_9['WeightRange'] = np.full((top_weight_8_9['WineName'].count(), 1), "0.8 <= Weight < 0.9")

        for i, r in top_weight_8_9.iterrows():
            df = df.append(r.to_dict(), ignore_index=True)

        top_weight_75_8 = out_df.query('Similarity >= 0.75 & Similarity < 0.8')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head()
        top_weight_75_8['Username'] = np.full((top_weight_75_8['WineName'].count(), 1), row['Username'])
        top_weight_75_8['WeightRange'] = np.full((top_weight_75_8['WineName'].count(), 1), "0.75 <= Weight < 0.8")

        for i, r in top_weight_75_8.iterrows():
            df = df.append(r.to_dict(), ignore_index=True)
        
        top_weight_5_75 = out_df.query('Similarity >= 0.5 & Similarity < 0.75')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head()
        top_weight_5_75['Username'] = np.full((top_weight_5_75['WineName'].count(), 1), row['Username'])
        top_weight_5_75['WeightRange'] = np.full((top_weight_5_75['WineName'].count(), 1), "0.5 <= Weight < 0.75")

        for i, r in top_weight_5_75.iterrows():
            df = df.append(r.to_dict(), ignore_index=True)

        # print('0.75 <= Similarity weight < 0.8')
        # print(out_df.query('Similarity >= 0.75 & Similarity < 0.8')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head())
        # print('0.5 <= Similarity weight < 0.75')
        # print(out_df.query('Similarity >= 0.5 & Similarity < 0.75')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head())

    return df.round(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process wine data.')
    parser.add_argument('product_csv', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='- Absolute path to Wine product csv file') 
    parser.add_argument('user_csv', type=argparse.FileType('r'), default=sys.stdin, help='- Absolute path to User csv file') 
    # parser.add_argument('--type', default='Both', help='- Wine Types: Red or White or Both')
    # parser.add_argument('--dry-sweet', type=float, default=0.0,help=' - Degree of Dry-Sweet. Ex: 1.0')
    # parser.add_argument('--light-bold', type=float, default=0.0,help=' - Degree of Light-Bold. Ex: 1.1')
    # parser.add_argument('--soft-acidic', type=float, default=0.0,help=' - Degree of Soft-Acidic. Ex: 1.2')
    # parser.add_argument('--smooth-tannic', type=float, default=0.0,help=' - Degree of Soft-Acidic. Ex: 1.3')
    args = parser.parse_args()
    # print(args)
    # user_wine_features = np.array([[args.dry_sweet, args.light_bold, args.soft_acidic, args.smooth_tannic]])
    # read csv file from first argument 
    df = pd.read_csv(args.product_csv)
    user_df = pd.read_csv(args.user_csv)
    
    report_df = compute_recommend_for_user(df,user_df)

    report_df.to_csv('results.csv')
   


