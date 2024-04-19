import os
import argparse
from tqdm import tqdm
from torchmetrics.text import WordErrorRate
from torch_geometric.loader import DataLoader
from torchmetrics.classification import BinaryF1Score
#
from utils import post_process
from networks import JAWSModel
from datasets import build_dataset

parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('model_path', type=str, help='Path to model weight file.')
parser.add_argument('dataset_path', type=str,
                    help='Path to dataset. (text file if of type khpos or directory if of type phylypo)')
parser.add_argument('--output_dir', type=str,
                    help='Path to output directory.', default=".")
parser.add_argument('--dataset_type', type=str,
                    help='The type of dataset to use',
                    default="khpos",
                    choices=['khpos', 'phylypo'])
args = parser.parse_args()

model = JAWSModel(args.config)
model.load(args.model_path)

dataset = build_dataset(args.dataset_path, args.dataset_type)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

wer, f1_score = WordErrorRate(), BinaryF1Score()

output_filename = os.path.basename(args.dataset_path)\
    .replace(".txt", ".result.txt")

with open(os.path.join(args.output_dir, output_filename), 'w') as outfile:
    total_wer, total_f1, n = 0, 0, 0
    for i, sample in enumerate(tqdm(dataloader)):
        characters, original = sample.characters[0], sample.original[0]

        y_pred = model.predict_sample(sample)
        out = post_process(y_pred.numpy(), characters)

        swer = wer([out], [original])
        f1 = f1_score(y_pred, sample.y.argmax(dim=-1))

        outfile.write(f"{swer}\t\t{original}\t\t{out}\n")

        total_wer += swer
        total_f1 += f1
        n += 1

    metrics = f"\nf1: {(total_f1/n)*100}, wer: {(total_wer/n)*100}"
    outfile.write(metrics)
    print(metrics)
