import requests
import ast
import json
import numpy as np
import pickle
import shutil
import os
import pandas as pd
import json


import warnings
warnings.filterwarnings("ignore")
infantads_subindustry = ['Food / Beverages - Baby Food','Health - Baby / Infant Care Products','Cosmetic / Hygiene - Baby / Infant Products']

dep_names = ['active_seconds', 'visible_seconds', 'attention_seconds','coview_seconds_active']
viewer_char = ['hh_id_hashed', 'demo_id_hashed', 'viewer_id_hashed','guest_flag']
ads_char = ['ad_id','industry','sub_industry','ad_url','airing_duration', 'creative_length','ad_type','viewing_start_time_utc']
program_char = ['content_id','program_genre','program_other_genres','program_type','new_episode']
context = ['channel_type','position_in_pod']

add_0929 = ['program_block_duration', 'program_time_start_utc','program_time_end_utc','viewing_end_time_utc']
add_1004 = ['timeshifted']
add_03212024 = ['airing_time_start_utc','airing_time_end_utc','unique_pod','frequency_attention','ad_title']

usecols = dep_names+viewer_char+ads_char+program_char+context+add_0929+add_1004+add_03212024

selected_categories = [
    'Entertainment / Media / Leisure',
    'Health',
    'Retail',
    'Legal / Financial',
    'Food / Beverages',
    'Automotive / Vehicles',
    'Cosmetic / Hygiene',
    'Government / Organizations',
    'Telecommunications',
    'Travel',
    'Home and Garden',
    'Electronics / Technology',
    'Education',
    'Apparel / Footwear / Accessories',
    'Restaurants',
    'Consumer Products'
]
corresponding_programs = {
    'Entertainment / Media / Leisure': ['entertainment', 'variety', 'reality', 'comedy', 'talk show', 'game show', 'talk', 'satire', 'music', 'pop culture'],
    'Health': ['health', 'medical', 'wellness', 'medical drama', 'lifestyle', 'fitness'],
    'Retail': ['shopping', 'consumer', 'retail', 'fashion', 'style', 'home projects', 'DIY', 'lifestyle', 'product showcase'],
    'Legal / Financial': ['legal', 'financial', 'business', 'finance', 'economics',],
    'Food / Beverages': ['cooking', 'culinary', 'food', 'culinary competition', 'cooking show', 'restaurant'],
    'Automotive / Vehicles': ['automotive', 'cars', 'auto racing', 'car show', 'travel', 'adventure'],
    'Cosmetic / Hygiene': ['beauty', 'cosmetic', 'hygiene', 'wellness', 'lifestyle', 'makeover'],
    'Government / Organizations': ['politics', 'government', 'news', 'current affairs', 'public affairs','documentary'],
    'Telecommunications': ['technology', 'science', 'tech news', 'telecommunications', 'innovation'],
    'Travel': ['travel', 'adventure', 'exploration', 'reality', 'lifestyle'],
    'Home and Garden': ['home projects', 'DIY', 'gardening', 'lifestyle', 'home improvement', 'real estate'],
    'Electronics / Technology': ['technology', 'innovation', 'science', 'tech news', 'gadgets'],
    'Education': ['education', 'documentary', 'knowledge', 'learning', 'science'],
    'Apparel / Footwear / Accessories': ['fashion', 'style', 'shopping', 'lifestyle', 'runway', 'clothing'],
    'Restaurants':['cooking', 'culinary', 'food', 'culinary competition', 'cooking show', 'restaurant'],
    'Consumer Products':['shopping', 'consumer', 'retail', 'fashion', 'style', 'home projects', 'DIY', 'lifestyle', 'product showcase']
}

#all_corresponding_programs = [genre for genres in corresponding_programs.values() for genre in genres]
all_corresponding_programs = set([genre for genres in corresponding_programs.values() for genre in genres])


def download_file(url, save_path):
    try:
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved to {save_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def preprocessing(df):
    #df['industry'] = df['industry'].replace('Restaurants','Food / Beverages')
    #df['industry'] = df['industry'].replace('Consumer Products','Retail')
    #df['program_other_genres'] = df['program_other_genres'].apply(ast.literal_eval)
    def determine_instream(row):
            if row['viewing_end_time_utc'] <= row['program_time_start_utc']:
                return 'preroll'
            elif row['viewing_start_time_utc'] >= row['program_time_end_utc']:
                return 'postroll'
            else:
                return 'midroll'
                
    def selection_midroll(df):
        df=df[df['instream']=='midroll']
        return df

    def selection_midpod(df):
        df['position_in_pod'] = df['position_in_pod'].str.lower()
        df=df[df['position_in_pod']!='first'] #consider all first, mid, last ad pods, edited 0929, this is different from pre/mid/post roll ads
        return df
        
    def selection_industry_genre(df):
        def check_genre_congruence(row):
            return any(genre in all_corresponding_programs for genre in row['program_other_genres'])
        #genre_congruence_mask = df.apply(check_genre_congruence, axis=1)
        #df = df[(df['industry'].isin(selected_categories)) & genre_congruence_mask & (df['program_genre'].isin(all_corresponding_programs))]
        
        df = df[(df['industry'].isin(selected_categories)) & (df['program_genre'].isin(all_corresponding_programs))]
        return df
        
    def selection_bad_rows(df):
        integer_rows_mask = df['program_other_genres'].apply(lambda x: isinstance(x, int))
        df = df[~integer_rows_mask]
        return df

    def selection_live_viewing(df):
        df=df[(df['timeshifted']==-1)|(df['timeshifted']=="-1")]
        return df

    def selection_airing_ad_duration(df):
        df = df[df['airing_duration']==df['creative_length']]
        return df
        
    def selection_airing_ad_viewing(df):
        #ad viewing start time should after the airing start time
        df = df[df['airing_time_start_utc']<=df['viewing_start_time_utc']] 
        # ad viewing end time should after the airing start time
        df = df[df['viewing_end_time_utc'] > df['airing_time_start_utc']]
        return df

    def unknow_error(df):
        df = df[df['creative_length']>=df['attention_seconds']]
        return df
    #df = selection_bad_rows(df)
    #df = selection_midpod(df)
    #
    df = selection_live_viewing(df)
    
    df['instream'] = df.apply(determine_instream, axis=1)
    df = selection_midroll(df)
    df = selection_airing_ad_duration(df) #airing duration equals to ad duration
    df = selection_airing_ad_viewing(df) #ad viewing start time no before than airing start time
    df = unknow_error(df)
    return selection_industry_genre(df)


def process_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', usecols=usecols)
        # df['program_other_genres'] = df['program_other_genres'].apply(convert_to_list)
        df = preprocessing(df)
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def infantads(df):
    df = df[df['sub_industry'].isin(infantads_subindustry)]
    return df

def find_max_in_sequences(data, min_gap=3):
    """
    Find the maximum value in each sequence of non-zero values, where a sequence is defined as a series of non-zero values separated by less than 3 consecutive zeros.
    
    Args:
        data (list): The input list of values.
        min_gap (int): The minimum number of consecutive zeros required to consider a new sequence.
        
    Returns:
        list: A list of tuples, where each tuple contains the index and value of the maximum in a sequence.
    """
    max_in_sequences = []
    in_segment = False
    prev_value = 0
    max_value = 0
    max_index = 0
    zero_count = 0
    
    for i, value in enumerate(data):
        if value > 0:
            if not in_segment or value > prev_value:
                in_segment = True
                prev_value = value
                max_value = value
                max_index = i
            else:
                if value > max_value:
                    max_value = value
                    max_index = i
            zero_count = 0
        else:
            zero_count += 1
            if zero_count >= min_gap:
                if in_segment:
                    max_in_sequences.append((max_index, max_value))
                in_segment = False
                prev_value = 0
                max_value = 0
                max_index = 0
    
    if in_segment:
        max_in_sequences.append((max_index, max_value))
    
    return max_in_sequences


def write_json_row(path, row):
    """
    Writes a single JSON row to a file.
    
    Args:
        path (str): The path to the output JSON file.
        row (dict): The dictionary containing the data for a single row.
    """
    
    for key, value in row.items():
        if isinstance(value, np.float32):
            row[key] = str(value)
            
    with open(path, 'a') as jsonfile:
        json.dump(row, jsonfile)
        jsonfile.write('\n')
