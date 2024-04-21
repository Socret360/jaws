# Just Another Word Segmentor (JAWS)

A Khmer word segmentation model based on Graph Neural Networks. It represent pairs of characters as nodes and perform node classification to determined if there is a space between two characters.

### Samples taken from Facebook posts

```
Source: តែបើអត់លុយសងផង មានលុយប្តឹងដែល ?
Output: តែ បើ អត់ លុយ សង ផង មាន លុយ ប្តឹង ដែល ?

Source: កូនជាប់រៀនទាំងអស់គ្នាមិនបានជូនដើរលេងទេ តែគ្មានអ្វីសប្បាយជាងការជួបជុំទេ
Output: កូន ជាប់ រៀន ទាំងអស់ គ្នា មិន បាន ជូន ដើរ លេង ទេ តែ គ្មាន អ្វី សប្បាយ ជាង ការ ជួប ជុំ ទេ

Source: ស្អាតណាស់ក្មេងតូច
Output: ស្អាត ណាស់ ក្មេង តូច

Source: ចង់ចូលផ្តល់កម្លាំងចិត្តនិងជួយស្រង់ក្លិននិងណាបង
Output: ចង់ ចូល ផ្តល់ កម្លាំង ចិត្ត និង ជួយ ស្រង់ ក្លិន និង ណា បង

Source: កាន់តស្រស់ស្អាតប្លែកតែម្ដងហើយ BA លៀសហាលយើងហ្នឹង
Output: កាន់ ត ស្រស់ ស្អាត ប្លែក តែ ម្ដង ហើយ លៀស ហាល យើង ហ្នឹង

Source: ខ្ញុំកំពុងនៅជាមួយខាងCMG CCTV ដើម្បីថតផ្សព្វផ្សាយវប្បធម៌ខ្មែរទៅកាន់ពិភពលោក ឥឡូវនេះខ្ញុំកំពុងមើលព្រះអាទិត្យរះ នៅប្រាសាទអង្គរវត្ត🇰🇭✨🌅
Output: ខ្ញុំ កំពុង នៅ ជាមួយ ខាង ដើម្បី ថត ផ្សព្វផ្សាយ វប្បធម៌ ខ្មែរ ទៅ កាន់ ពិភព លោក ឥឡូវ នេះ ខ្ញុំ កំពុង មើល ព្រះ អាទិត្យ រះ នៅ ប្រាសាទ អង្គរ វត្ត
```

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

# Future Improvements

- [ ] Handling of non-Khmer Characters

  The current version of JAWS strip away characters that are not in the VOCAB list. An improvement to the usability of the model would be to take into account non-Khmer characters that are in the input text. One way to do this, I believe, is to include an addition element in the character vector to represent the unknown characters that are not in the VOCAB list.

- [ ] Determining Prefix, Suffix, Compound, Spaces, and Non-Space

  The current implementation runs a binary node classification. A more useful extension would be to modify the task to a multi-class classification where the model classify if two characters are a prefix, a suffix, a compound, a space, or no space at all.

- [ ] Khmer Character Cluster (KCC) Violation Penalty

  The current model sometimes outputs separators that cause words to break their correct spelling. I believe, there are two ways to handle this. One, additional element can be added to the input node vector to signify if it is part of a character clusters or not. This will ensure that nodes that are made up of characters that formed a KCC never get separated. Another way, is to add an another loss term to the current loss function that penalize predictions that violates the KCC convention.
