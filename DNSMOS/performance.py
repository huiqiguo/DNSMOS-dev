# Usage:
# python performance.py .\csv\readspeech.csv

import argparse
import pandas as pd

def main(args):
    df = pd.read_csv(args.filepath)
    ovrScore = df['OVRL']
    groundTruth = df['P808_MOS']
    mse = round(((groundTruth - ovrScore)**2).mean(), 5)
    print(f'Dataset: {args.filepath}, MSE = {mse}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath',
                        help='csv file path for analysis')
    
    args = parser.parse_args()

    main(args)