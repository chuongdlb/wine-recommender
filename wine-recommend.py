
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

def recommend_using_cosine_similarity(df, user_wine_features):
    print(df[['Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic']])
    df['Similarity'] = cosine_similarity(df[['Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic']], user_wine_features)
    df['VivinoRating'] = df['VivinoRating'] / df['Price']
    df['BlendedScore'] = df['Similarity'] * .5 + df['VivinoRating'] * .5
    print(df)
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
    print(user_wine_features)
    # read csv file from first argument 
    df = pd.read_csv(args.csvfile)
    
    print(df)
    p_df = preprocess(df, args.type)
    print(p_df[['WineName', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic']])

    out_df = recommend_using_cosine_similarity(p_df, user_wine_features)
    out_df.head()
    print(out_df[['WineName', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic', 'VivinoRating', 'BlendedScore', 'Similarity']].head())