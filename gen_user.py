import datetime as dt
import pandas as pd
import numpy as np
# from numpy.random import randn
import argparse
import sys
# from pandas_datareader import data, wb
from faker.providers.person.en import Provider
import random
from scipy import stats


def random_names(name_type, size):
    """
    Generate n-length ndarray of person names.
    name_type: a string, either first_names or last_names
    """
    names = getattr(Provider, name_type)
    return np.random.choice(names, size=size)

def generate_user_mock_df(size = 10):
    df = pd.DataFrame(columns=['Username', 'Type', 'Dry-Sweet', 'Light-Bold', 'Soft-Acidic', 'Smooth-Tannic'])
    df['Username'] = random_names('first_names', size)
    df['Type'] = np.random.choice(['White', 'Red', 'Both'], size=size)
    df['Dry-Sweet'] =  np.random.uniform(low=0.5, high=9.0, size=(size,))
    df['Light-Bold'] =  np.random.uniform(low=0.5, high=9.0, size=(size,))
    df['Soft-Acidic'] =  np.random.uniform(low=0.5, high=9.0, size=(size,))
    df['Smooth-Tannic'] = np.random.uniform(low=0.5, high=9.0, size=(size,))
    return df

if __name__ == "__main__":
    df = generate_user_mock_df(10)
    df.to_csv("user.csv",index=False)