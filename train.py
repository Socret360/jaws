import os
import argparse
#
from utils import VOCAB
from networks import JAWSModel
from datasets import build_dataset


parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('dataset_path', type=str,
                    help='Path to dataset. (text file if of type khpos or directory if of type phylypo)')
parser.add_argument('--output_dir', type=str,
                    help='Path to output directory.', default=".")
parser.add_argument('--dataset_type', type=str,
                    help='The type of dataset to use',
                    default="khpos",
                    choices=['khpos', 'phylypo'])
parser.add_argument('--lr', type=float,
                    help='The learning rate',
                    default=0.01)
parser.add_argument('--early_stopping_patience', type=int,
                    help='The learning rate',
                    default=10)
parser.add_argument('--epochs', type=int,
                    help='The number of epochs to train',
                    default=None)
parser.add_argument('--batch_size', type=int,
                    help='The number of samples per batch',
                    default=32)
args = parser.parse_args()


dataset = build_dataset(
    args.dataset_path,
    args.dataset_type
)


model = JAWSModel(args.config)

model.fit(
    data=dataset,
    epochs=args.epochs,
    learning_rate=args.lr,
    batch_size=args.batch_size,
    model_temp_dir=args.output_dir,
    early_stopping_patience=args.early_stopping_patience,
)

output_filename = os.path.basename(args.config).replace(".json",  ".pt")
model.save(os.path.join(args.output_dir, output_filename))
