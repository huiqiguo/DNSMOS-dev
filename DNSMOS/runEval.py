# This script runs the performance.py script to compute the MSE for a given set of csv files

import os
import subprocess

# Insert names of csv files to analyse
# Files should be stored inside a subfolder named 'csv'
files = [
    'readspeech.csv',
    'readspeech_personalised.csv',
    'emotionalspeech.csv',
    'emotionalspeech_personalised.csv',
    'vocalset48khzmono.csv',
    'vocalset48khzmono_personalised.csv'
]

for file in files:
    path = os.path.join('.\csv', file)
    subprocess.run(['python', 'performance.py', path])