import os
import json
import torch
import tempfile
from tqdm import tqdm
from typing import List, Dict
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv
from torchmetrics.classification import BinaryF1Score
from kornia.losses import binary_focal_loss_with_logits
from torchmetrics.text import CharErrorRate, WordErrorRate
#
from utils import pretty_training_log, text_to_graph, post_process, read_config_file, VOCAB


class JAWSNetwork(torch.nn.Module):
    def __init__(
            self,
            num_features: int,
            hidden_dims: List[int],
            type: str = "gcn",
    ):
        super().__init__()

        conv = {
            "gcn": GCNConv,
            "gat": GATConv,
        }[type]
        all_dims = [num_features] + hidden_dims
        self.layers = ModuleList([
            conv(all_dims[i], all_dims[i+1])
            for i in range(len(all_dims)-1)
        ])
        self.predictor = conv(sum(hidden_dims), 2)

    def forward(self, data):
        h0, edge_index = data.x, data.edge_index
        #
        out = [h0]
        for i, layer in enumerate(self.layers):
            h = layer(out[i], edge_index).relu()
            out.append(h)

        y = self.predictor(torch.concat(out[1:], dim=-1), edge_index)
        out = F.softmax(y, dim=1)
        return out


class JAWSModel:
    def __init__(self, config_file: str) -> None:
        config = read_config_file(config_file)
        self.model = JAWSNetwork(
            type=config["type"],
            num_features=len(VOCAB)*2,
            hidden_dims=config["hidden_dims"],
        )

    def fit(
            self,
            data,
            val_data=None,
            batch_size=32,
            learning_rate=0.01,
            mode: str = "min",
            monitor: str = "loss",
            early_stopping_patience: int = 10,
            restore_best_weights: int = True,
            logging_interval: int = 1,
            epochs: int = None,
            model_temp_dir: str = None,
    ):
        self.temp_weights_path = os.path.join(
            tempfile.gettempdir() if model_temp_dir is None else model_temp_dir,
            "best_weights.pt"
        )
        self.f1_score = BinaryF1Score()
        self.wer = WordErrorRate()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        has_val_set = val_data is not None
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        val_loader = None
        if has_val_set:
            val_loader = DataLoader(
                val_data,
                shuffle=False,
                batch_size=batch_size,
            )

        self.best_epoch, best_metric, epoch = 0, None, 0

        while True:

            # check if maximum epoch reached or early stopping criterion reached
            reached_max_epochs = epochs is not None and epoch >= epochs
            exceeded_early_stop_patience = (
                epoch - self.best_epoch) > early_stopping_patience if early_stopping_patience is not None else False

            if reached_max_epochs or exceeded_early_stop_patience:
                break

            self.model.train(True)

            epoch_result = self.__train(train_loader, use_grad=True)

            if has_val_set:
                self.model.train(False)
                epoch_result_val = self.__train(val_loader, use_grad=False)
                epoch_result = {
                    **epoch_result,
                    **{f"val_{k}": v for k, v in epoch_result_val.items()}
                }

            curr_metric = epoch_result[monitor]

            has_improved = True
            if best_metric is not None:
                if mode == "min":
                    has_improved = curr_metric < best_metric
                elif mode == "max":
                    has_improved = curr_metric > best_metric

            if has_improved:
                best_metric = curr_metric
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.temp_weights_path)

            training_log = {
                "epoch": epoch,
                "best_epoch": self.best_epoch,
                **epoch_result
            }

            if epoch % logging_interval == 0:
                pretty_log = pretty_training_log(training_log)
                print(pretty_log)

            epoch += 1

        if restore_best_weights:
            self.model.load_state_dict(torch.load(self.temp_weights_path))

    def save(self, output_path: str):
        """ Save the model weights to `output_path`.

        Args:
        ---
            - output_path: str - The path to the output file.
        """
        torch.save(self.model.state_dict(), output_path)

    def load(self, model_path: str):
        """ Load the weights from `model_path`.

        Args:
        ---
            - model_path: str
                The path to the model file.
        """
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))

    def segment(self, text: str) -> str:
        """ Segment the `text` into words.

        Args:
        ---
            - text: str - The input text.

        Returns:
        ---
            output segmented text.
        """
        data = text_to_graph(text)
        o = self.model(data)
        y_pred = o.argmax(dim=1).numpy()
        out = post_process(y_pred, data.characters)
        return out

    def predict_sample(self, sample: Data) -> torch.Tensor:
        o = self.model(sample)
        return o.argmax(dim=1)

    def __train(self, dataloader: DataLoader, use_grad=False) -> Dict[str, float]:
        total_loss, total_f1, total_wer, n = 0, 0, 0, 0

        for data in tqdm(dataloader):
            if use_grad:
                self.optimizer.zero_grad()

            out = self.model(data)
            text = self.__prediction_to_string(
                out.argmax(dim=-1),
                data.batch,
                data.characters,
            )
            loss = binary_focal_loss_with_logits(
                out, data.y, alpha=0.75, gamma=2.0, reduction='mean')

            f1 = self.f1_score(out.argmax(dim=-1), data.y.argmax(dim=-1))
            wer = self.wer(text, data.original)

            total_loss += loss
            total_f1 += f1
            total_wer += wer
            n += 1

            if use_grad:
                loss.backward()
                self.optimizer.step()

        return {
            "loss": float(total_loss/n),
            "f1": float(total_f1/n),
            "wer": float(total_wer/n),
        }

    def __prediction_to_string(self, y_preds, batch, characters):
        out = []
        for i in range(len(characters)):
            text = post_process(
                y_preds[batch == i],
                characters[i]
            )
            out.append(text)
        return out
