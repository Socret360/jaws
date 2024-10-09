# Just Another Word Segmentor (JAWS)

A Khmer word segmentation model based on Graph Neural Networks. It represent pairs of characters as nodes and perform node classification to determined if there is a space between two characters.

### Samples taken from Facebook posts

```
Source: តែបើអត់លុយសងផង មានលុយប្តឹងដែល ?
Output: តែ បើ អត់លុយ សង ផង មាន លុយ ប្តឹង ដែល ?

Source: កូនជាប់រៀនទាំងអស់គ្នាមិនបានជូនដើរលេងទេ តែគ្មានអ្វីសប្បាយជាងការជួបជុំទេ
Output: កូន ជាប់ រៀន ទាំង អស់ គ្នា មិន បានជូន ដើរ លេង ទេ តែ គ្មាន អ្វី សប្បាយ ជាង ការជួប ជុំទេ

Source: ស្អាតណាស់ក្មេងតូច
Output: ស្អាត ណាស់ ក្មេង តូច

Source: ចង់ចូលផ្តល់កម្លាំងចិត្តនិងជួយស្រង់ក្លិននិងណាបង
Output: ចង់ ចូល ផ្តល់ កម្លាំង ចិត្ត និង ជួយ ស្រង់ ក្លិន និង ណាបង

Source: កាន់តស្រស់ស្អាតប្លែកតែម្ដងហើយ BA លៀសហាលយើងហ្នឹង
Output: កាន់ តស្រស់ ស្អាត ប្លែក តែ ម្ដង ហើយ BA លៀស ហាល យើង ហ្នឹង

Source: ខ្ញុំកំពុងនៅជាមួយខាងCMG CCTV ដើម្បីថតផ្សព្វផ្សាយវប្បធម៌ខ្មែរទៅកាន់ពិភពលោក ឥឡូវនេះខ្ញុំកំពុងមើលព្រះអាទិត្យរះ នៅប្រាសាទអង្គរវត្ត🇰🇭✨🌅
Output: ខ្ញុំ កំពុង នៅ ជាមួយ ខាង CMG CCTV ដើម្បី ថត ផ្សព្វផ្សាយ វប្បធម៌ ខ្មែរ ទៅ កាន់ ពិភព លោក ឥឡូវ នេះ ខ្ញុំ កំពុង មើល ព្រះ អាទិត្យ រះ នៅ ប្រាសាទ អង្គរ វត្ត 🇰🇭✨🌅

Source: សូម្បីតែសម្លេងក៏ឈ្លោះគ្នាដែរ ឈ្លោះគ្រប់យ៉ាង mak Lin 😭
Output: សូម្បី តែ សម្លេង ក៏ ឈ្លោះ គ្នា ដែរ ឈ្លោះ គ្រប់ យ៉ាង mak Lin 😭

Source: Global Green Growth Week បានត្រលប់មកវិញហើយ នៅសប្តាហ៍ក្រោយនេះ!
Output: Global Green Growth Week បាន ត្រលប់ មក វិញ ហើយ នៅ សប្តាហ៍ ក្រោយ នេះ !

Source: ញាត្តិគាំទ្រ Cambodian Pageant និងអ្នកគាំទ្រវិស័យបវរកញ្ញាកម្ពុជាចំពោះប្រសាសន៍លោកស្រីអ៉ឹម សុគន្ធា Sokunthea Im ក្នុងការប្តេជ្ញាចិត្ត សំណូមពរឳ្យស្ថាប័នពាក់ព័ន្ធ បិទការប្រកួត និងការបញ្ជូនបេក្ខភាពតំណាងកម្ពុជាចូលរួមប្រកួត Miss Grand International
Output: ញា ត្តិ គាំទ្រ Cambodian Pageant និង អ្នក គាំទ្រ វិស័យ បវរ កញ្ញា កម្ពុជា ចំពោះ ប្រសាសន៍ លោក ស្រី អ៉ឹម សុគន្ធា Sokunthea Im ក្នុង ការ ប្តេជ្ញា ចិត្ត សំណូម ពរឳ្យ ស្ថាប័ន ពាក់ ព័ន្ធ បិទ ការ ប្រកួត និង ការ បញ្ជូន បេក្ខ ភាព តំណាង កម្ពុជា ចូល រួម ប្រកួត Miss Grand International
```

## How to run

### Segment

You can perform word segmentation using the pretrained model by running the below command.

```bash
usage: segment.py [-h] [--file_mode] [--console_output] config model_path sample

Run the segmentation on one sample text file.

positional arguments:
  config            Path to config file.
  model_path        Path to model weight file.
  sample            Path to the input text file containing the text to segment.

optional arguments:
  -h, --help        show this help message and exit
  --file_mode       Wether sample is a file
  --console_output  Whether to output the result to console
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
| JAWSGCN | 98.035% | 5.915% |
| JAWSGAT | 98.462% | 4.911% |

**Close Set** [here](https://github.com/ye-kyaw-thu/khPOS/blob/master/corpus-draft-ver-1.0/data/CLOSE-TEST.word)
| Model | F1 Score | Word Error Rate |
| ------- | -------- | --------------- |
| JAWSGCN | 98.814% | 3.855% |
| JAWSGAT | 98.941% | 3.487% |
