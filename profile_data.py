import os
import sys

sys.path.append('.')
sys.path.append('..')

os.environ['TMPDIR'] = os.environ.get('OUTPUT_DATA')
os.environ['MPLCONFIGDIR'] = os.environ.get('OUTPUT_DATA')

import glob
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    # name of just file, not full path
    last_slash = list(find_all(all_files[0], '/'))[-1]
    file_name = all_files[0][last_slash:]

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

# Helper function to create numeric data frame
def create_numeric_df(metadata_df, df):
    # filter the data frame for just the numeric columns
    numeric_columns = metadata_df[metadata_df['Read-in Type'] != 'object']

    # filter for numeric columns
    numeric_df = df.loc[:, numeric_columns['Variable'].to_numpy()]

    return numeric_df

# Numeric Variable Exploration: Do not use variable type guesses
def numeric_exploration1(metadata_df, df):
    # grab data frame for only numeric values
    numeric_df = create_numeric_df(metadata_df, df)

    # Loop through all the numeric columns for descriptive statistics
    numerical_list = []
    for column in numeric_df.columns:
        subset_data = numeric_df.loc[:, column]
        numerical_list.append(subset_data.describe().round(2).apply(lambda x: format(x, 'f')).str.slice(stop = -4))
    numerical_list = pd.concat(numerical_list, axis = 1)

    return numerical_list

# Create Correlations Images
# 1. Pearson
def pearson_correlation(metadata_df, df):
    cmap = sns.diverging_palette(10, 150, as_cmap = True)

    # grab data frame for only numeric values
    numeric_df = create_numeric_df(metadata_df, df)
    
    # create correlation matrices and corresponding heatmaps
    corr_matrix = numeric_df.corr().round(2)
    plt.figure()
    pearson_plot = sns.heatmap(corr_matrix,
                               annot = True,
                               cmap = cmap,
                               square = True)
    plt.savefig(os.environ.get('OUTPUT_DATA') + '/pearson_corr.png',
                facecolor = 'w')

# 2. Kendall
def kendall_correlation(metadata_df, df):
    cmap = sns.diverging_palette(10, 150, as_cmap = True)

    # grab data frame for only numeric values
    numeric_df = create_numeric_df(metadata_df, df)
    
    # create correlation matrices and corresponding heatmaps
    corr_matrix = numeric_df.corr(method = 'kendall').round(2)
    plt.figure()
    kendall_plot = sns.heatmap(corr_matrix,
                               annot = True,
                               cmap = cmap,
                               square = True)
    plt.savefig(os.environ.get('OUTPUT_DATA') + '/kendall_corr.png',
                facecolor = 'w')

# 3. Spearman
def spearman_correlation(metadata_df, df):
    cmap = sns.diverging_palette(10, 150, as_cmap = True)

    # grab data frame for only numeric values
    numeric_df = create_numeric_df(metadata_df, df)
    
    # create correlation matrices and corresponding heatmaps
    corr_matrix = numeric_df.corr(method = 'spearman').round(2)
    plt.figure()
    spearman_plot = sns.heatmap(corr_matrix,
                                annot = True,
                                cmap = cmap,
                                square = True)
    plt.savefig(os.environ.get('OUTPUT_DATA') + '/spearman_corr.png',
                facecolor = 'w')

'''
# Output the names, file extensions, other text info
def text_info(number_of_files, extension, file_name,
              number_of_rows, number_of_columns):
    line1 = 'Files in directory: ' + '{:,}'.format(number_of_files)
    line2 = 'File extension: ' + str(extension)
    line3 = 'File name: ' + str(file_name)
    line4 = 'Observations: ' + '{:,}'.format(number_of_rows)
    line5 = 'Variables: ' + '{:,}'.format(number_of_columns)

    lines = [line1, line2, line3, line4, line5]

    with open(os.environ.get('OUTPUT_DATA') + '/results.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
'''

def create_html(metadata_df, number_of_files, extension, file_name,
                number_of_rows, number_of_columns, numeric_expl_df):
    # Save variables as correctly formatted strings
    metadata_df_html = metadata_df.to_html(index = False, justify = 'center')
    numeric_df_html = numeric_expl_df.to_html(index = True, justify = 'center')

    # Create and open the file
    html_file = open(os.environ.get('OUTPUT_DATA') + '/Profile.html', 'w')

    # Write to the HTML file
    html_file.write('''<html>
    <head>
    <style>
    .myDiv {
        border: 1px outset black;
        text-align: left;
    }
    </style>
    <title>HTML File</title>
    </head>
    <body>
    <h1> Data Profile </h1>
    </body>
    <html>''')

    html_file.write('<div class="myDiv"><h2>Overview</h2><p>Files in directory: {code}<br>'.format(code = '{:,}'.format(number_of_files)))
    html_file.write('File extension: {code}<br>'.format(code = extension))
    html_file.write('File name: {code}<br>'.format(code = file_name))
    html_file.write('Observations: {code}<br>'.format(code = '{:,}'.format(number_of_rows)))
    html_file.write('Variables: {code}</p></div><br>'.format(code = '{:,}'.format(number_of_columns)))
    html_file.write('<div class="myDiv"><h2>Variable Overview</h2>')
    html_file.write(metadata_df_html)
    html_file.write('</div><br><div class="myDiv"><h2>Numerical Variable Descriptive Statistics</h2>')
    html_file.write(numeric_df_html)
    html_file.write('</div>')

    html_file.close()

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

    # Numeric Exploration
    numeric_expl_df = numeric_exploration1(metadata_df, df)

    # Image Outputs
    pearson_correlation(metadata_df, df)
    kendall_correlation(metadata_df, df)
    spearman_correlation(metadata_df, df)

    logging.info(f'Finished Profiling.')

    # Outputs
    # metadata_df.to_csv(os.environ.get('OUTPUT_DATA') + '/metadata.csv')
    # text_info(number_of_files, extension, file_name,
    #           number_of_rows, number_of_columns)
    create_html(metadata_df, number_of_files, extension, file_name,
                number_of_rows, number_of_columns, numeric_expl_df)

    logging.info(f'Data profile output written.')

if __name__ == '__main__':
    main()