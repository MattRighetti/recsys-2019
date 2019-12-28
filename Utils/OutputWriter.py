import os
import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np

def write_output(fittedRecommender, target_user_list):
    """
    Create a new file and writes to it
    :param fittedRecommender: recommender already fitted
    :param target_user_list: list of target users
    :return:
    """
    file = open(create_unique_file(), "w+")

    print("Getting recommendations...")
    rec_list = fittedRecommender.recommend(target_user_list, cutoff=10)

    user_index = 0
    df_list = []

    print("Generating new list...")

    for list_ in tqdm(rec_list):
        row = [target_user_list[user_index], np.squeeze(np.array(list_))]
        df_list.append(row)
        user_index += 1

    cols = ['user_id', 'item_list']

    df = pd.DataFrame(df_list, columns=cols)

    string = df.to_csv(index=False)
    string = string.replace(']', '')
    string = string.replace('[', '')

    file.write(string)

    file.close()
    print("Success")

def create_unique_file():
    """
    Creates filename
    :return: New random filename file_out_day_hour_minute_second.csv
    """
    current_date = datetime.datetime.now()
    folder_path = "/Users/mattiarighetti/Developer/PycharmProjects/recsys/output/"

    file_name = f'file_out_{current_date.day}_{current_date.hour}_{current_date.minute}_{current_date.second}.csv'

    print(file_name)

    return os.path.join(folder_path, file_name)