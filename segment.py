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

args = parser.parse_args()

model = JAWSModel(args.config)
model.load(args.model_path)

with open(args.sample, 'r') as input_file:
    preprocessed, _ = preprocess_phylypo_sample(input_file.read())
    out = model.segment(preprocessed)

with open(args.sample.replace(".txt", ".seg.txt"), "w") as outfile:
    outfile.write(out)
