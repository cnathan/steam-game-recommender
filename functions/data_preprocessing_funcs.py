import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import sys
import IPython
import psutil
import os


import utility_funcs as util

def load_data(dir):

  # Read in data files
  df_games = pd.read_csv(dir + 'games.csv')
  df_users = pd.read_csv(dir + 'users.csv')
  df_recs = pd.read_csv(dir + 'recommendations.csv')
  df_games_meta = pd.read_json(dir + 'games_metadata.json', lines=True, orient="records")

  # Perform a left join on the games and games_meta dataframes using the 'app_id' column
  df_games = df_games.merge(df_games_meta, on='app_id', how='left')

  # Rename certain columns for readability
  df_games.rename(columns = {'positive_ratio':'rating_positive_ratio', 'user_reviews':'game_review_count'}, inplace = True)
  df_users.rename(columns = {'products':'games_played', 'reviews':'user_review_count'}, inplace = True)
  df_recs.rename(columns = {'hours':'hours_played'}, inplace = True)
  
  # Organize dataframes into a dictionary for easy access
  steam_data = {'games': df_games, 'users': df_users, 'recs': df_recs}
      
  return steam_data


def unpack_dict(data_dict):
  return data_dict['games'], data_dict['users'], data_dict['recs']







def print_rating_cats(df_games):
    
  # Build dataframe of categories
  cat_counts_df = df_games['rating'].value_counts().reset_index().rename(columns={'index': 'rating', 'rating': 'count'})

  # Define the desired sorting order
  SORT_ORDER  = ['Overwhelmingly Positive', 'Very Positive', 'Mostly Positive', 'Positive', 'Mixed', 
                  'Negative', 'Mostly Negative', 'Very Negative', 'Overwhelmingly Negative']

  # Create a categorical data type with the specified order
  cat_dtype = pd.CategoricalDtype(categories=SORT_ORDER, ordered=True)

  # Convert the 'rating' column to the categorical data type
  cat_counts_df['rating'] = cat_counts_df['rating'].astype(cat_dtype)

  # Sort the DataFrame by the 'rating' column
  sorted_df = cat_counts_df.sort_values(by='rating')

  # Print unique categories sorted best-to-worst
  print(f'There are {len(cat_counts_df)} rating categories:')
  display(sorted_df.style.hide(axis='index'))

  return
    




import matplotlib.pyplot as plt

import matplotlib.pyplot as plt



import matplotlib.pyplot as plt



def calculate_density(df):
  n_users = df['user_id'].nunique()
  n_items = df['app_id'].nunique()
  n_non_zero_elements = df.shape[0]
  density = n_non_zero_elements / (n_users * n_items)
  density_percentage = density * 100
  print(f'User-Item Matrix Density (%): {round(density_percentage,4)}%\n')
  return


def print_stats(df_games, df_users, df_recs, filtered_games, filtered_users, filtered_recs):
    perc_games_remaining = len(filtered_games) / len(df_games) * 100
    perc_users_remaining = len(filtered_users) / len(df_users) * 100
    perc_recs_remaining = len(filtered_recs) / len(df_recs) * 100

    print(f"Games remaining (%):\t {perc_games_remaining:.2f}%")
    print(f"Users remaining (%):\t {perc_users_remaining:.2f}%")
    print(f"Reviews remaining (%):\t {perc_recs_remaining:.2f}%\n")
    calculate_density(filtered_recs)
    return


def filter_1(data_orig, data, verbose=True):
    
    # Separate dataframes for easier access within funcion
    df_games, df_users, df_recs = data_orig['games'], data_orig['users'], data_orig['recs']
    df_games_pre, df_users_pre, df_recs_pre = data['games'], data['users'], data['recs']
    
    # Find user_ids of users with zero playtime
    user_ids_zero_playtime = df_recs_pre[df_recs_pre['hours_played'] == 0]['user_id'].unique()

    # Filter df_users, df_games, and df_recs
    df_games_filt = df_games_pre
    df_recs_filt = df_recs_pre[~df_recs_pre['user_id'].isin(user_ids_zero_playtime)]
    df_users_filt = df_users_pre[~df_users_pre['user_id'].isin(user_ids_zero_playtime)]
    
    if verbose:
        print(f"Filter 1: Keep Reviews from Users with more than 0 hours of playtime\n")
        print_stats(df_games, df_users, df_recs, df_games_filt, df_users_filt, df_recs_filt)
    
    # Join dataframes back into dictionary (easier for passing between functions)
    steam_data_filt = {'games': df_games_filt, 'users': df_users_filt, 'recs': df_recs_filt}
    
    return steam_data_filt
        

def filter_2(data_orig, data, verbose=True):
    
    # Separate dataframes for easier access within funcion
    df_games, df_users, df_recs = data_orig['games'], data_orig['users'], data_orig['recs']
    df_games_pre, df_users_pre, df_recs_pre = data['games'], data['users'], data['recs']
    
    # Find user_ids of users with zero products
    user_ids_zero_products = df_users_pre[df_users_pre['games_played'] == 0]['user_id'].unique()

    # Filter df_users, df_games, and df_recs
    df_games_filt = df_games_pre
    df_recs_filt = df_recs_pre[~df_recs_pre['user_id'].isin(user_ids_zero_products)]
    df_users_filt = df_users_pre[~df_users_pre['user_id'].isin(user_ids_zero_products)]
    
    if verbose:
        print(f"Filter 2: Keep Reviews from Users with more than 0 Games\n")
        print_stats(df_games, df_users, df_recs, df_games_filt, df_users_filt, df_recs_filt)

    # Join dataframes back into dictionary (easier for passing between functions)
    steam_data_filt = {'games': df_games_filt, 'users': df_users_filt, 'recs': df_recs_filt}
    
    return steam_data_filt
        

def filter_3(data_orig, data, threshold=60000, verbose=True, show_plots=True):
    
    # Separate dataframes for easier access within funcion
    df_games, df_users, df_recs = data_orig['games'], data_orig['users'], data_orig['recs']
    df_games_pre, df_users_pre, df_recs_pre = data['games'], data['users'], data['recs']
    
    # Calculate the number of reviews for each game and cumulative percentage of games
    game_review_counts = df_recs_pre['app_id'].value_counts().sort_values(ascending=False)
    cumulative_percent_games = np.cumsum(game_review_counts) / game_review_counts.sum() * 100

    # Plot the cumulative percentage of games against the number of reviews per game
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(game_review_counts.values, cumulative_percent_games)
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.xlabel('Number of Reviews per Game')
        plt.ylabel('Cumulative Percentage of Games')
        plt.title('Elbow Method for Optimal Threshold of Reviews per Game')
        plt.show()

    # Filter out games and reviews based on the filtered game IDs
    df_games_filt = df_games_pre[df_games_pre['game_review_count'] >= threshold]
    df_users_filt = df_users_pre
    df_recs_filt = df_recs_pre[df_recs_pre['app_id'].isin(df_games_filt['app_id'])]
    
    # Filter to only include games with threshold or more reviews
    game_review_counts = df_recs_filt.groupby('app_id').size()
    games_to_keep = game_review_counts[game_review_counts >= threshold].index    
    df_recs_filt = df_recs_filt[df_recs_filt['app_id'].isin(games_to_keep)]
    
    if verbose:
        print(f"Filter 3: Keep Reviews for Games with more than {threshold} reviews\n")
        print_stats(df_games, df_users, df_recs, df_games_filt, df_users_filt, df_recs_filt)
        print(f"Minimum number of reviews per game: {df_recs_filt.groupby('app_id').size().min()}\n")

    # Join dataframes back into dictionary (easier for passing between functions)
    steam_data_filt = {'games': df_games_filt, 'users': df_users_filt, 'recs': df_recs_filt}
    
    return steam_data_filt
        

def filter_4(data_orig, data, threshold=100, verbose=True, show_plots=True):
    
    # Separate dataframes for easier access within funcion
    df_games, df_users, df_recs = data_orig['games'], data_orig['users'], data_orig['recs']
    df_games_pre, df_users_pre, df_recs_pre = data['games'], data['users'], data['recs']
    
    # Sort and plot the number of users vs the number of games/products
    df_users_sorted = df_users_pre.sort_values(by='games_played', ascending=False).reset_index(drop=True)

    # Plot the number of users vs the number of games/products
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(df_users_sorted.index, df_users_sorted['games_played'])
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.title('Number of Users vs Number of Games Played')
        plt.xlabel('Users')
        plt.ylabel('Games Played')
        plt.xlim(0, len(df_users_sorted)/1000)
        plt.ylim(0, df_users_sorted['games_played'].max())
        plt.show()

    # Filter df_users, df_games, and df_recs
    df_games_filt = df_games_pre
    df_users_filt = df_users_pre[df_users_pre['games_played'] >= threshold]
    df_recs_filt = df_recs_pre[df_recs_pre['user_id'].isin(df_users_filt['user_id'])]
    
    if verbose:
        print(f"Filter 4: Keep Reviews from Users with more than {threshold} games\n")
        print_stats(df_games, df_users, df_recs, df_games_filt, df_users_filt, df_recs_filt)
        print(f"Minimum number of games per user: {df_users_filt['games_played'].min()}\n")

    # Join dataframes back into dictionary (easier for passing between functions)
    steam_data_filt = {'games': df_games_filt, 'users': df_users_filt, 'recs': df_recs_filt}
    
    return steam_data_filt
        

def filter_5(data_orig, data, threshold=5, verbose=True, show_plots=True):
    
    # Separate dataframes for easier access within funcion
    df_games, df_users, df_recs = data_orig['games'], data_orig['users'], data_orig['recs']
    df_games_pre, df_users_pre, df_recs_pre = data['games'], data['users'], data['recs']

    # Sort users by the number of reviews
    df_users_sorted = df_users_pre.sort_values(by='user_review_count', ascending=False).reset_index(drop=True)

    # Plot the number of users vs number of reviews per user
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(df_users_sorted['user_review_count'])
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Elbow Point ({threshold})')
        plt.title('Number of Users vs Number of Reviews per User')
        plt.xlabel('User Index')
        plt.ylabel('Number of Reviews')
        plt.legend()
        plt.xlim(0, 200)
        plt.show()

    # Filter df_users, df_games, and df_recs
    df_games_filt = df_games_pre
    df_users_filt = df_users_pre[df_users_pre['user_review_count'] >= threshold]
    df_recs_filt = df_recs_pre[df_recs_pre['user_id'].isin(df_users_filt['user_id'])]

    # Group by user_id and count the number of reviews for each user
    user_review_counts = df_recs_filt.groupby('user_id').size()

    # Filter out users with less than threshold reviews
    users_to_keep = user_review_counts[user_review_counts >= threshold].index

    # Filter the df_recs_filt DataFrame to only include users with threshold or more reviews
    df_recs_filt = df_recs_filt[df_recs_filt['user_id'].isin(users_to_keep)]
    
    if verbose:
        print(f"Filter: Keep Reviews from Users with more than {threshold} Reviews\n")
        print_stats(df_games, df_users, df_recs, df_games_filt, df_users_filt, df_recs_filt)
        print(f"Minimum number of reviews for any user: {df_recs_filt.groupby('user_id').size().min()}\n")
        
    # Join dataframes back into dictionary (easier for passing between functions)
    steam_data_filt = {'games': df_games_filt, 'users': df_users_filt, 'recs': df_recs_filt}
    
    return steam_data_filt
        


def filter_6(data_orig, data, perc=1, verbose=True):
    # Separate dataframes for easier access within function
    df_games, df_users, df_recs = data_orig['games'], data_orig['users'], data_orig['recs']
    df_games_pre, df_users_pre, df_recs_pre = data['games'], data['users'], data['recs']

    # Get user-ids with more than one review
    users_with_multiple_reviews = df_recs_pre.groupby('user_id').filter(lambda x: len(x) > 1)['user_id'].unique()

    # Randomly sample user-ids based on the percentage specified
    sampled_user_ids = np.random.choice(users_with_multiple_reviews, int(len(users_with_multiple_reviews) * perc), replace=False)

    # Filter the dataframes
    df_recs_filt = df_recs_pre[df_recs_pre['user_id'].isin(sampled_user_ids)]
    df_games_filt = df_games_pre[df_games_pre['app_id'].isin(df_recs_filt['app_id'])]
    df_users_filt = df_users_pre[df_users_pre['user_id'].isin(df_recs_filt['user_id'])]

    if verbose:
        print(f"Filter: Randomly Keep {perc*100}% of Users with More than One Review\n")
        print_stats(df_games, df_users, df_recs, df_games_filt, df_users_filt, df_recs_filt)

    # Join dataframes back into dictionary (easier for passing between functions)
    steam_data_filt = {'games': df_games_filt, 'users': df_users_filt, 'recs': df_recs_filt}

    return steam_data_filt


def preprocess_data(data, thresh = [60000,100,5,1], verbose=True, plot=False):

    if verbose:
        print("#####################")
        print("####  FILTERING  ####")
        print("#####################\n")
    data_filt = data.copy()
    data_filt = filter_1(data, data_filt, verbose=verbose) # Drop Reviews from Users with Zero Playtime
    data_filt = filter_2(data, data_filt, verbose=verbose) # Drop Reviews from Users with Zero Games
    data_filt = filter_3(data, data_filt, threshold = thresh[0], verbose=verbose, show_plots=plot) # Drop Reviews for Games with Few Players/Reviews
    data_filt = filter_4(data, data_filt, threshold = thresh[1], verbose=verbose, show_plots=plot) # Drop Reviews from Users with Few Games
    data_filt = filter_5(data, data_filt, threshold = thresh[2], verbose=verbose, show_plots=plot) # Drop Reviews from Users with Few Reviews
    data_filt = filter_6(data, data_filt, perc = thresh[3], verbose=verbose)
    return data_filt


def get_users_items(df):
  
    # Find the total number of unique users and items
    unique_users = df['user_id'].nunique()
    unique_items = df['app_id'].nunique()

    return unique_users, unique_items
    

def plot_pdf(df_recs):

  # Count the number of unique app_id per user
  app_count_per_user = df_recs.groupby('user_id')['app_id'].nunique()

  # Plot the PDF vs. unique app_id
  plt.figure(figsize=(10, 5))
  sns.histplot(data=app_count_per_user, kde=False, bins=50, color='blue')
  plt.title('PDF vs. Unique app_id per User')
  plt.xlabel('Unique app_id per User')
  plt.ylabel('Probability Density Function (PDF)')
  plt.show()


def plot_playtime_distribution(df_recs):

  # Group data by playtime hours and count the number of users
  playtime_user_count = df_recs.groupby('hours_played')['user_id'].count().reset_index()

  # Plot the playtime distribution
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=playtime_user_count, x='hours_played', y='user_id')
  plt.title('Playtime Distribution: Number of Users vs Playtime')
  plt.xlabel('Playtime (in hours)')
  plt.ylabel('Number of Users')
  plt.show()


def plot_cdf(df_users):
  # Calculate the empirical CDF
  df_users_sorted = df_users_filt.sort_values(by='games_played')
  cumulative_prob = np.arange(1, len(df_users_sorted) + 1) / len(df_users_sorted)

  # Plot the empirical CDF vs number of reviews
  plt.figure(figsize=(10, 5))
  plt.plot(df_users_sorted['games_played'], cumulative_prob, color='blue')
  plt.title('Empirical CDF of Number of Reviews')
  plt.xlabel('Number of Reviews')
  plt.ylabel('Empirical CDF')
  plt.xlim(0, 5000)

  plt.show()



def print_system_memory_info():
    
    # Get the virtual memory object
    memory_info = psutil.virtual_memory()

    print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Memory usage percentage: {memory_info.percent}%\n")
    
    return memory_info

def estimate_memory_consumption(steam_data, test_size=0.2, rating_metric='is_recommended', verbose = True):
    
    df = steam_data['recs']
    
    memory_per_cell = {
        "int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "int32": 4, "uint32": 4,
        "int64": 8, "uint64": 8, "float16": 2, "float32": 4, "float64": 8,
        "complex64": 8, "complex128": 16, "object": 8, "bool": 1,
    }

    unique_users = round(len(df['user_id'].unique()) * (1-test_size))
    unique_items = round(len(df['app_id'].unique()) * (1-test_size))
    
    total_cells_user_item = unique_users * unique_items
    total_cells_user_user = unique_users * unique_users
    total_cells_item_item = unique_items * unique_items

    dtype_str = str(df[rating_metric].dtype)
    
    if dtype_str in memory_per_cell:
        
        total_memory_user_item = total_cells_user_item * memory_per_cell[dtype_str]
        total_memory_user_user = total_cells_user_user * memory_per_cell[dtype_str]
        total_memory_iem_item = total_cells_item_item * memory_per_cell[dtype_str]
        
        memory_gb_user_item = total_memory_user_item / (1024 ** 3)
        memory_gb_user_user = total_memory_user_user / (1024 ** 3)
        memory_gb_item_item = total_memory_iem_item / (1024 ** 3)
        
        if verbose:
            print("###############################")
            print("####  MEMORY REQUIREMENTS  ####")
            print("###############################\n")

            print(f"Estimated Memory Consumption for Training Set ({(1-test_size)*100}%):")
            print(f"User-Item Matrix: ({unique_users}, {unique_items}) => {total_cells_user_item} total cells => {memory_gb_user_item:.4f} GB")
            print(f"User-User Matrix: ({unique_users}, {unique_users}) => {total_cells_user_user} total cells => {memory_gb_user_user:.4f} GB")
            print(f"Item-Item Matrix: ({unique_items}, {unique_items}) => {total_cells_item_item} total cells => {memory_gb_item_item:.4f} GB")
        
        # Get the virtual memory object
        memory_info = psutil.virtual_memory()

        if memory_gb_user_user >= memory_info.total / (1024 ** 3):
            print('<<< WARNING >>>\nRequired memory may exceed total memory.\n')
            print_system_memory_info()
            user_input = input("Do you want to proceed? (y/n): ")
            if user_input.lower() == "n": raise HaltExecution("Stopping the execution.")
        elif memory_gb_user_user >= memory_info.available / (1024 ** 3):
            print('<<< WARNING >>>\nRequired memory may exceed available memory.\n')
            print_system_memory_info()
            user_input = input("Do you want to proceed? (y/n): ")
            if user_input.lower() == "n": raise HaltExecution("Stopping the execution.")

    return




