class DataWriter(object):
    def __init__(self, dirPath, fileName):
        self.dirPath = dirPath
        self.fileName = fileName

    def writeToOutput(self, recommendations_array):
        filePath = self.dirPath + self.fileName + ".csv"
        file = open(filePath, 'w+')
        file.write('user_id, item_list\n')

        counter = 0

        for user_array in recommendations_array:
            file.write(f'{counter}')
            for i in range(0, 9):
                if i != 9:
                    file.write(f'{user_array[i]}')
                else:
                    file.write(f'{user_array[i]}')
            break

        file.close()