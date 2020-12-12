'''
Short script used to examine the format of the data in a news timeline dataset
'''

import os
import pandas as pd
from date_models.model_utils import inspect

def main():
    input_file = '../datasets/t17/bpoil/articles.preprocessed_mod.jsonl'
    df = pd.read_json(input_file)
    inspect(df)



if __name__ == '__main__':
    print('Starting: ', os.path.basename(__file__))
    main()
    print('Completed: ', os.path.basename(__file__))
