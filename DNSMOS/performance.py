# Usage:
# python performance.py -f .\csv\readspeech.csv -n readspeech
# python performance.py -f .\csv\readspeech_p.csv -n readspeech -p
# python performance.py -f .\csv\vocalset48khzmono.csv -n vocalset48khzmono
# python performance.py -f .\csv\vocalset48khzmono_p.csv -n vocalset48khzmono -p

import argparse
import pandas as pd

def main(args):
    df = pd.read_csv(args.filepath)
    ovrScore = df['OVRL']
    groundTruth = df['P808_MOS']
    mse = round(((groundTruth - ovrScore)**2).mean(), 5)
    print(f'Dataset: {args.name}, Personalisation = {args.personalisation}, MSE = {mse}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath',
                        help='csv file path for analysis')
    parser.add_argument('-n', '--name',
                        help='Dataset name corresponding to the csv file')
    parser.add_argument('-p', '--personalisation', action='store_true',
                        help='Boolean value indicating whether personalisation was used')    
    
    args = parser.parse_args()

    main(args)