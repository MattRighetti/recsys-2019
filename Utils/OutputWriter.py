import os
import datetime

def write_output(fittedRecommender, target_user_list, test=True):
    """
    Create a new file and writes to it
    :param fittedRecommender: recommender already fitted
    :param target_user_list: list of target users
    :return:
    """
    file = open(create_unique_file(), "w+")
    file.write("user_id,item_list\n")

    for user_id in target_user_list:
        recommendations = fittedRecommender.recommend(user_id, 10)
        array_string = " ".join(str(x) for x in recommendations)
        file.write(f'{user_id},{array_string}\n')

    file.close()
    print("Success")

def create_unique_file(test=True):
    """
    Creates filename
    :return: New random filename file_out_day_hour_minute_second.csv
    """
    current_date = datetime.datetime.now()
    if test:
        folder_path = "../output"
    else:
        folder_path = "./output"

    file_name = f'file_out_{current_date.day}_{current_date.hour}_{current_date.minute}_{current_date.second}.csv'

    print(file_name)

    return os.path.join(folder_path, file_name)