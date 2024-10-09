import os
import argparse
#
from utils import preprocess_phylypo_sample
from networks import JAWSModel

parser = argparse.ArgumentParser(
    description='Run the segmentation on one sample text file.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('model_path', type=str, help='Path to model weight file.')
parser.add_argument('sample', type=str,
                    help='Path to the input text file containing the text to segment.')
parser.add_argument('--file_mode', action='store_true',
                    help='Wether sample is a file')
parser.add_argument('--console_output', action='store_true',
                    help='Whether to output the result to console')

args = parser.parse_args()

model = JAWSModel(args.config)
model.load(args.model_path)

if args.file_mode:
    with open(args.sample, 'r') as input_file:
        input_str = input_file.read()
else:
    input_str = args.sample

preprocessed, _ = preprocess_phylypo_sample(input_str)
out = model.segment(preprocessed)

if not args.console_output:
    with open(args.sample.replace(".txt", ".seg.txt"), "w") as outfile:
        outfile.write(out)
else:
    print(out)
