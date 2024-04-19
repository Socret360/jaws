# Just Another Word Segmentor (JAWS)

A Khmer word segmentation model based on Graph Neural Networks. It represent pairs of characters as nodes and perform node classification to determined if there is a space between two characters.

## How to run

### Segment

You can perform word segmentation using the pretrained model by running the below command.

```bash
usage: segment.py [-h] config model_path sample

Run the segmentation on one sample text file.

positional arguments:
  config      Path to config file.
  model_path  Path to model weight file.
  sample      Path to the input text file containing the text to segment.

optional arguments:
  -h, --help  show this help message and exit
```

### Training

The model can be retrained on your custom dataset using the below command.

```bash
usage: train.py [-h] [--output_dir OUTPUT_DIR] [--dataset_type {khpos,phylypo}] [--lr LR] [--early_stopping_patience EARLY_STOPPING_PATIENCE] [--epochs EPOCHS] config dataset_path

Run the training loop.

positional arguments:
  config                Path to config file.
  dataset_path          Path to dataset. (text file if of type khpos or directory if of type phylypo)

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Path to output directory.
  --dataset_type {khpos,phylypo}
                        The type of dataset to use
  --lr LR               The learning rate
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        The learning rate
  --epochs EPOCHS       The number of epochs to train
```

### Evaluation

To evaluate a trained model, use the below command.

```bash
usage: evaluate.py [-h] [--output_dir OUTPUT_DIR] [--dataset_type {khpos,phylypo}] config model_path dataset_path

Run the training loop.

positional arguments:
  config                Path to config file.
  model_path            Path to model weight file.
  dataset_path          Path to dataset. (text file if of type khpos or directory if of type phylypo)

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Path to output directory.
  --dataset_type {khpos,phylypo}
                        The type of dataset to use
```

## About Pretrained Weights

The pretrained weights are available [here](pretrained). Each weights were produced by training using the dataset from KhPOS available [here](https://github.com/ye-kyaw-thu/khPOS).
Below is the results on the test sets from the set repository.

**Open Set**: [here](https://github.com/ye-kyaw-thu/khPOS/blob/master/corpus-draft-ver-1.0/data/OPEN-TEST.word)
| Model | F1 Score | Word Error Rate |
| ------- | -------- | --------------- |
| JAWSGCN | 96.831% | 9.077% |
| JAWSGAT | 97.814% | 6.871% |

**Close Set** [here](https://github.com/ye-kyaw-thu/khPOS/blob/master/corpus-draft-ver-1.0/data/CLOSE-TEST.word)
| Model | F1 Score | Word Error Rate |
| ------- | -------- | --------------- |
| JAWSGCN | 97.433% | 7.651% |
| JAWSGAT | 98.236% | 5.221% |
