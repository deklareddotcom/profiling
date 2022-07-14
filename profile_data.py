import os
import sys

sys.path.append('.')
sys.path.append('..')

os.environ['TMPDIR'] = os.environ.get('OUTPUT_DATA')
os.environ['MPLCONFIGDIR'] = os.environ.get('OUTPUT_DATA')

import glob
import logging
import numpy as np
import pandas as pd

# read in the data
# output consolidated DF, number of files
def read(input_folder):
    # list files
    all_files = glob.glob(input_folder + '/*')

    # count files
    number_of_files = len(all_files) - 1

    # extension of the file
    last_period = list(find_all(all_files[0], '.'))[-1]
    extension = all_files[0][last_period:]
    file_name = all_files[0]

    # pull the data together
    li = []
    for filename in all_files:
        if filename.endswith('_SUCCESS'):
            continue
        df = pd.read_csv(filename, header = 0)
        li.append(df)
    frame = pd.concat(li, axis = 0)

    return frame, number_of_files, extension, file_name

# function to find all matching characters in a string
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)

# metadata - unique values, data types, etc.
def metadata(df):
    column_types = df.dtypes.reindex(index = df.columns)
    null_values = df.isnull().sum()
    
    # combine the variable read types and number of nulls
    df_admin = pd.concat([column_types, null_values], axis = 1).reset_index()
    df_admin.columns = ['Variable', 'Read-in Type', 'Nulls']
    column_names = df.columns

    # find unique values, etc.
    var_list = []
    for column in column_names:
        column_of_interest = df[column]
        unique_values = column_of_interest.nunique()
        total_values = column_of_interest.count()

        # difference between total & unique values
        unique_total_diff = total_values - unique_values
        unique_total_quot = unique_total_diff / total_values * 100
        pct_unique = str(round(100 - unique_total_quot, 1)) + '%'

        # Guess cat/bool/continuous variable type
        if unique_values == 2:
            var_type = 'Boolean'
        elif unique_total_diff == 0:
            var_type = 'ID'
        elif total_values <= 50:
            threshold = 70
            if unique_total_quot > threshold:
                var_type = 'Categorical'
            else:
                var_type = 'Continuous'
        else:
            threshold = 90
            if unique_total_quot > threshold:
                var_type = 'Categorical'
            else:
                var_type = 'Continuous'

        var_info = pd.DataFrame([[column,
                                  var_type,
                                  unique_values,
                                  total_values,
                                  unique_total_diff,
                                  pct_unique]])
        var_list.append(var_info)
    
    # concatenate and clean
    var_list = pd.concat(var_list)
    var_list.columns = ['Variable',
                        'Type',
                        'Unique',
                        'total_values',
                        'difference',
                        'Percent Unique']

    # combine the data frames
    full_metadata = pd.merge(df_admin,
                             var_list.drop(['total_values', 'difference'],
                                            axis = 1),
                             on = 'Variable')
    return full_metadata

# Numeric Variable Exploration: Do not use variable type guesses
def numeric_exploration1(metadata_df, df):
    # filter the data frame for just the numeric columns
    numeric_columns = metadata_df[metadata_df['Read-in Type'] != 'object']

    # filter for numeric columns
    numeric_df = df.loc[:, numeric_columns['Variable'].to_numpy()]

    # Loop through all the numeric columns for descriptive statistics
    numerical_list = []
    for column in numeric_df.columns:
        subset_data = numeric_df.loc[:, column]
        numerical_list.append(subset_data.describe())
    numerical_list = pd.concat(numerical_list, axis = 1)

    return numerical_list

# Main function for Docker
def main():

    # setup logging handler
    logs_directory = os.environ.get('HABU_CONTAINER_LOGS')
    log_file = f'{logs_directory}/container.log'
    logging.basicConfig(
        handlers = [logging.FileHandler(filename = log_file,
                                        encoding = 'utf-8',
                                        mode = 'a+')],
                    format = '%(asctime)s %(name)s:%(levelname)s:%(message)s',
                    datefmt = '%F %A %T',
                    level = logging.INFO,
    )

    logging.info(f'Start Processing...')
    
    # load the data
    input_location = os.environ.get('INPUT_DATA')
    logging.info(f'Reading data in from {input_location}')
    
    df, number_of_files, extension, file_name = read(os.environ.get('INPUT_DATA'))

    logging.info(f'Data successfully read.')
    logging.info(f'Read in {number_of_files} files.')
    logging.info(f'Beginning to profile...')

    # Profiling the data
    number_of_rows, number_of_columns = df.shape

    # Variable Types
    metadata_df = metadata(df)

    logging.info(f'Finished Profiling.')

    metadata_df.to_csv(os.environ.get('OUTPUT_DATA') + '/metadata.csv')

    logging.info(f'Data profile output written.')

if __name__ == '__main__':
    main()