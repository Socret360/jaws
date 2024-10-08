import re
import json
import torch
import itertools
import numpy as np
from typing import Dict, Any
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import Dataset

NON_SPACE_SEP = "</NS>"
SPACE_SEP = "</SP>"
CONSTS = list(set(u'កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ'))
VOWELS = list(set(u'឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8'))
SUB = list(set(u'្'))
DIAC = list(set(u"\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0"))
SYMS = list(set('៕។៛ៗ៚៙៘,.? '))
NUMBERS = list(set(u'០១២៣៤៥៦៧៨៩0123456789'))
LUNAR = list(set('᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿'))
VOCAB = sorted(CONSTS + VOWELS + SUB + DIAC + NUMBERS + SYMS + LUNAR)
REGEX_MATCHES_ALL_IGNORE_SPACES = f"([^ {''.join(re.escape(char) for char in VOCAB)}]+)(\u0020)|(\u0020)([^ {''.join(re.escape(char) for char in VOCAB)}]+)"
# including unk characters
FEATURE_VECTOR_LENGTH = len(VOCAB)+2

CHAR_TO_INT = {v: i for i, v in enumerate(VOCAB)}
CHAR_TO_INT["[SP]"] = len(VOCAB)

LABEL_MAP = [NON_SPACE_SEP, SPACE_SEP]


def preprocess_khpos_sample(text: str):
    """ Convert `text` in each line of the KhPos dataset to
    the appropriate format. (e.g. char</NS>char</SP>)

    Args:
    ---
        - `text`: str
            The input text to preprocess.

    Returns:
    ---
        (str, str)
            A tuple containing the preprocessed text and the raw clean input text.
    """
    text = text.strip('\n')
    # set compound words, prefix, and suffix marker as space
    text = re.sub('[\^~_]', '\u0020', text)
    preprocesed, text = preprocess_phylypo_sample(text)
    return preprocesed, text


def preprocess_phylypo_sample(text: str) -> str:
    """ Convert `text` in each sample of the Phylypo dataset to
    the appropriate format. (e.g. char</NS>char</SP>)

    Args:
    ---
        - `text`: str
            The input text to preprocess.

    Returns:
    ---
        (str, str)
            A tuple containing the preprocessed text and the raw clean input text.
    """
    text = text.strip('\n')
    text = text.strip()
    text = text.replace('️', '\u200b')
    text = text.replace('\n', '\u0020')
    text = re.sub('\u0020+', '\u0020', text)
    text = re.sub(
        rf"([^{re.escape(''.join(VOCAB))}])\u0020",
        lambda match: f"{match.group(1)}[SP]",
        text
    )
    text = re.sub(
        rf"\u0020([^{re.escape(''.join(VOCAB))}])",
        lambda match: f"[SP]{match.group(1)}",
        text
    )
    text = re.sub(
        rf"([{re.escape(''.join(VOCAB))}])([^{re.escape(''.join(VOCAB))}])",
        lambda match: f"{match.group(1)}[SP]{match.group(2)}",
        text
    )
    text = re.sub(
        rf"([^{re.escape(''.join(VOCAB))}])([{re.escape(''.join(VOCAB))}])",
        lambda match: f"{match.group(1)}[SP]{match.group(2)}",
        text
    )
    text = re.sub(r'(\[SP\])+', "[SP]", text)
    preprocessed = NON_SPACE_SEP.join(re.findall(r'\[SP\]|.', text))
    preprocessed = preprocessed.replace(
        f"{NON_SPACE_SEP}\u0020{NON_SPACE_SEP}",
        SPACE_SEP
    )
    return preprocessed, text.replace("[SP]", "\u0020")


def text_to_graph(text: str, original: str = None):
    """ Convert `text` in a graph representation suitable for training.

    Args:
    ---
        - `text`: str
            The preprocessed input text. (e.g. char</NS>char</SP>
        - `original`: str
            The original clean representation of the text.

    Returns:
    ---
        str
            The graph representation of `text`.
    """
    delimeter = f"{NON_SPACE_SEP}|{SPACE_SEP}"
    characters = re.split(delimeter, text)
    separators = re.findall(delimeter, text)

    features = np.array([CHAR_TO_INT[char] if char in CHAR_TO_INT.keys() else FEATURE_VECTOR_LENGTH-1
                         for char in characters])

    characters = "".join(characters)
    characters = characters.replace("[SP]", '\u0020')
    characters = list(characters)

    labels = np.array([LABEL_MAP.index(sep) for sep in separators])

    x = np.zeros((len(characters), FEATURE_VECTOR_LENGTH))
    x[np.arange(0, len(characters)), features] = 1
    x = np.repeat(x, 2, axis=0)[1:-1]
    x = x.flatten().reshape((len(separators), FEATURE_VECTOR_LENGTH*2))

    y = np.zeros((len(separators), len(LABEL_MAP)))
    y[np.arange(0, len(separators)), labels] = 1

    edge_index = np.expand_dims(np.arange(0, x.shape[0]-1), axis=-1)
    edge_index = np.concatenate([
        edge_index,
        edge_index+1,
    ], axis=1)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    edge_index = torch.from_numpy(edge_index).t().contiguous()

    graph = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        characters=characters,
        separators=separators,
        original=original,
    )
    graph = T.ToUndirected()(graph)

    return graph


def post_process(
    y_pred: torch.tensor,
    characters: np.array,
) -> str:
    """ Convert a single model prediction to output text.

    Args:
    ---
        - `y_pred`: torch.tensor
            The predicted tensor of shape (num_nodes,) produced by the model.
        - `characters`: np.array
            A numpy array containing all the characters in the input text.

    Returns:
        str
            The output segmented text.
    """
    y_pred = list(np.array(LABEL_MAP)[y_pred])+[None]
    out = "".join(list(itertools.chain(
        *zip(characters,  y_pred)))[:-1])
    out = out.replace(NON_SPACE_SEP, '')
    out = out.replace(SPACE_SEP, '\u0020')
    return out


def pretty_training_log(log: Dict[str, Any]):
    """ Produce a prettry training log.

    Args:
    ---
        - `log`: Dict[str, Any]
            The training log produced during each epoch of training.

    Returns:
    ---
        str
            String of `log` separated by commas.
    """
    output = []
    for k, v in log.items():
        output.append(f"{k}: {v}")

    return ", ".join(output)


def read_config_file(filepath: str) -> Dict[str, Any]:
    """ Read the config file.

    Args:
    ---
        - `filepath`: str
            Path to the config file.

    Returns:
    ---
        Dict[str, Any]
            Dictionary representing the configuration file.
    """
    with open(filepath, "r") as file:
        return json.load(file)
