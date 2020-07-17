
import datetime as dt
import pandas as pd
import numpy as np
# from numpy.random import randn
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
# from pandas_datareader import data, wb

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process wine data.')
    parser.add_argument('csvfile', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='- Absolute path to csv file') 
    parser.add_argument('--type', default='Both', help='- Wine Types: Red or White or Both')
    parser.add_argument('--dry-sweet', type=float, default=0.0,help=' - Degree of Dry-Sweet. Ex: 1.0')
    parser.add_argument('--light-bold', type=float, default=0.0,help=' - Degree of Light-Bold. Ex: 1.1')
    parser.add_argument('--soft-acidic', type=float, default=0.0,help=' - Degree of Soft-Acidic. Ex: 1.2')
    parser.add_argument('--smooth-tannic', type=float, default=0.0,help=' - Degree of Soft-Acidic. Ex: 1.3')
    

    args = parser.parse_args()
    print(args)
    user_wine_features = np.array([[args.dry_sweet, args.light_bold, args.soft_acidic, args.smooth_tannic]])
    # read csv file from first argument 
    df = pd.read_csv(args.csvfile)
    # pre-process
    p_df = preprocess(df, args.type)
    # normalize Vivino score
    p_df = normalize_vivino_score(p_df)
    # print(p_df[['WineName', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic']])

    out_df = recommend_using_cosine_similarity(p_df, user_wine_features)
    print('Similarity weight >= 0.9')
    print(out_df[out_df['Similarity'] >= 0.9][['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head())
    print('0.8 <= Similarity weight < 0.9')
    print(out_df.query('Similarity >= 0.8 & Similarity < 0.9')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head())
    print('0.75 <= Similarity weight < 0.8')
    print(out_df.query('Similarity >= 0.75 & Similarity < 0.8')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head())
    print('0.5 <= Similarity weight < 0.75')
    print(out_df.query('Similarity >= 0.5 & Similarity < 0.75')[['WineName', 'Vintage', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head())


