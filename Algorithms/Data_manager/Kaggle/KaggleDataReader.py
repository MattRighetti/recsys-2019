import zipfile, shutil
import pandas as pd

from Algorithms.Data_manager.DataReader import DataReader
from Algorithms.Data_manager.DataReader_utils import downloadFromURL, merge_ICM


def _loadURM_preinitialized_item_id(filePath, header = False, separator=",",
                                     if_new_user = "add", if_new_item = "ignore",
                                     item_original_ID_to_index = None,
                                     user_original_ID_to_index = None):


    from Algorithms.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    URM_all_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = item_original_ID_to_index,
                                                    on_new_col = if_new_item,
                                                    preinitialized_row_mapper = user_original_ID_to_index,
                                                    on_new_row = if_new_user)

    if header:
        df_original = pd.read_csv(filepath_or_buffer=filePath, sep=separator, header= 0 if header else None,
                        usecols=['row', 'col', 'data'],
                        dtype={'row':str, 'col':str, 'data':float})
    else:
        df_original = pd.read_csv(filepath_or_buffer=filePath, sep=separator, header= 0 if header else None,
                        dtype={0:str, 1:str, 2:float})

        df_original.columns = ['row', 'col', 'data']

    # Remove data with rating non valid
    df_original.drop(df_original[df_original.data == 0.0].index, inplace=True)

    user_id_list = df_original['row'].values
    item_id_list = df_original['col'].values
    rating_list = df_original['data'].values

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)

    return  URM_all_builder.get_SparseMatrix(), \
            URM_all_builder.get_column_token_to_id_mapper(), \
            URM_all_builder.get_row_token_to_id_mapper()





def _loadICM_subclass(ICM_path, header=True, separator=','):

    # Genres
    from Algorithms.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(ICM_path, "r", encoding="latin1")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if numCells % 1000000 == 0:
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            item_id = line[0]

            feature = line[1]

            # Rows item ID
            # Cols features
            ICM_builder.add_single_row(item_id, feature, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()




def _loadUCM(UCM_path, header=True, separator=','):

    # Genres
    from Algorithms.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(UCM_path, "r", encoding="latin1")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if numCells % 1000000 == 0:
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            user_id = line[0]

            feature = line[1]

            # Rows user ID
            # Cols features
            ICM_builder.add_single_row(user_id, feature, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()






class KaggleDataReader(DataReader):

    DATASET_SPLIT_ROOT_FOLDER = "/Users/mattiarighetti/Developer/PycharmProjects/recsys/Algorithms/Data_manager/"
    DATASET_OFFLINE_ROOT_FOLDER = "/Users/mattiarighetti/Developer/PycharmProjects/recsys/Algorithms/Data_manager/"

    # This subfolder contains the preprocessed data, already loaded from the original data file
    DATASET_SUBFOLDER_ORIGINAL = "original/"

    DATASET_URL = "https://www.kaggle.com/c/15854/download-all"
    DATASET_SUBFOLDER = "Kaggle/dataset_zip/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_subclass"]
    AVAILABLE_UCM = ["UCM_age", "UCM_region"]
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original

        print("KaggleReader: Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        print("------->" + zipFile_path)

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "data.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("KaggleReader: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "data.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "data.zip")


        ICM_asset_path = dataFile.extract("data_ICM_asset.csv", path=zipFile_path + "decompressed/")
        ICM_price_path = dataFile.extract("data_ICM_price.csv", path=zipFile_path + "decompressed/")
        ICM_subclass_path = dataFile.extract("data_ICM_sub_class.csv", path=zipFile_path + "decompressed/")
        UCM_region_path = dataFile.extract("data_UCM_region.csv", path=zipFile_path + "decompressed/")
        UCM_age_path = dataFile.extract("data_UCM_age.csv", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("data_train.csv", path=zipFile_path + "decompressed/")

        ICM_subclass, tokenToFeatureMapper_ICM_subclass, self.item_original_ID_to_index = _loadICM_subclass(ICM_subclass_path,
                                                                                                     header=True,
                                                                                                     separator=',')

        self._LOADED_ICM_DICT["ICM_subclass"] = ICM_subclass
        self._LOADED_ICM_MAPPER_DICT["ICM_subclass"] = tokenToFeatureMapper_ICM_subclass

        ######
        self.UCM_region, self.tokenToFeatureMapper_UCM_region, self.user_original_ID_to_index = _loadUCM(UCM_region_path,
                                                                                                   header=True,
                                                                                                   separator=',')

        self.UCM_age, self.tokenToFeatureMapper_UCM_age, _ = _loadUCM(UCM_age_path, header=True, separator=',')

        ######

        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = _loadURM_preinitialized_item_id(
            URM_path, separator=",",
            header=True, if_new_user="add", if_new_item="ignore",
            item_original_ID_to_index=self.item_original_ID_to_index, user_original_ID_to_index=self.user_original_ID_to_index)

        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        print("KaggleData: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("KaggleData: saving URM")


if __name__ == '__main__':
    data = KaggleDataReader()
    data.load_data(False)
    data.get_loaded_ICM_dict()