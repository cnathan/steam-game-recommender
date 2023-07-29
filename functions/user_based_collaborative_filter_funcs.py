import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
import IPython
import psutil


import utility_funcs as util
import data_preprocessing_funcs as fn




from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def split_user_data(user_data, test_size=0.2):
    # Split the data into train and test sets
    train_data, test_data = train_test_split(user_data, test_size=test_size, random_state=42)
    
    # # Further split the train data into train and validation sets
    # train_data, val_data = train_test_split(train_data, test_size=test_size/(1-test_size), random_state=42)
    
    # return train_data, val_data, test_data
    return train_data, test_data


import pandas as pd
from sklearn.model_selection import train_test_split
def stratified_split(df, target_col, test_size=0.2, random_state=None):
    # Split the data into train and test sets
    train_data, test_data = train_test_split(df, stratify=df[target_col], test_size=test_size, random_state=random_state)
    
    return train_data, test_data




from scipy.sparse import csr_matrix

def get_users_items(df):
  
    # Find the total number of unique users and items
    unique_users = df['user_id'].nunique()
    unique_items = df['app_id'].nunique()

    return unique_users, unique_items

def get_sparsity(df):
  
    # Find the total number of unique users and items
    unique_users, unique_items = get_users_items(df)

    # Calculate the total number of possible user-item pairs
    total_user_item_pairs = unique_users * unique_items

    # Find the number of non-zero values in the dataset
    non_zero_values = df[df['hours_played'] != 0].shape[0]

    # Calculate the number of zero values
    zero_values = total_user_item_pairs - non_zero_values

    # print("Number of zero values:", zero_values)

    sparsity = 1 - (non_zero_values / total_user_item_pairs)
    sparsity_percentage = sparsity * 100

    print(f"User-Item Matrix sparsity: {sparsity_percentage:.2f}%")



def estimate_csr_matrix_memory(df):
    
    # Find the total number of unique users and items
    num_users, unique_items = get_users_items(df)
    num_non_zero_elements = df[df['is_recommended'] != 0].shape[0]

    indices_memory = 4 * num_non_zero_elements
    indptr_memory = 4 * (num_users + 1)
    data_memory = 4 * num_non_zero_elements
    
    total_memory = indices_memory + indptr_memory + data_memory
    total_memory_mb = total_memory / (1024 ** 2)
    total_memory_gb = total_memory / (1024 ** 3)
    
    return total_memory_mb, total_memory_gb





from sklearn.metrics.pairwise import cosine_similarity
def get_predicted_rating(user_id, item_id, predicted_ratings, user_index, item_index):
    try:
        # Find the positions of the user and item in the ratings matrix
        user_position = np.where(user_index == user_id)[0][0]
        item_position = np.where(item_index == item_id)[0][0]
        # Return the predicted rating
        return predicted_ratings[user_position, item_position]
    except IndexError:
        # Return NaN if the user or item was not found in the ratings matrix
        return np.nan

def predict_ratings(user_similarity, ratings):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = ratings - mean_user_rating[:, np.newaxis]
    
    # Calculate the predicted ratings based on user similarities and normalized ratings
    pred = user_similarity.dot(ratings_diff) / (np.array([np.abs(user_similarity).sum(axis=1)]).T + 1e-8) # small val to avoid div by zero
    
    # Add the mean user rating back to get the final predicted ratings
    return pred + mean_user_rating[:, np.newaxis]



import matplotlib.pyplot as plt


from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score


import seaborn as sns
def plot_distribution(low_error_counts, high_error_counts, xlabel, ylabel, title):
    low_error_df = pd.DataFrame(low_error_counts).reset_index().rename(columns={'index': xlabel, xlabel: 'low_error'})
    high_error_df = pd.DataFrame(high_error_counts).reset_index().rename(columns={'index': xlabel, xlabel: 'high_error'})
    merged_df = low_error_df.merge(high_error_df, on=xlabel, how='outer').fillna(0)
    
    # Plot the distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=xlabel, y='low_error', data=merged_df, color='blue', ax=ax, label='Low Error')
    sns.barplot(x=xlabel, y='high_error', data=merged_df, color='red', ax=ax, label='High Error')
    ax.legend()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.show()



def compute_summary_statistics(counts):
    stats = {
        'min': counts.min(),
        'max': counts.max(),
        'mean': counts.mean(),
        'median': counts.median(),
        'std_dev': counts.std(),
        'q25': counts.quantile(0.25),
        'q75': counts.quantile(0.75),
    }
    return stats


import pandas as pd


