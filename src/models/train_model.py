""" Module contains functions for training models """

import argparse
import copy
import math
import os
import warnings

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

AVAILABLE_MODELS = ["bart", "custom_transformer"]
AVAILABLE_DATASET_SIZES = ["lg", "md", "sm", "xs"]


### Classes for CustomTransformer Network


class Logger:
    """Manage log messages"""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def log(self, message: str):
        """Log message to console

        Args:
            message (str): message to log
        """
        if self.verbose:
            print(message)


### Classes for CustomTransformer Network


class PositionalEncoding(nn.Module):
    """Add positional encoding"""

    def __init__(self, embedding_size: int, dropout: float, max_size: int):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            -torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size
        )
        pos = torch.arange(0, max_size).reshape(max_size, 1)
        pos_embedding = torch.zeros((max_size, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        """Make forward pass"""
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    """Learn embedding"""

    def __init__(self, vocab_size: int, embedding_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, tokens: Tensor):
        """Make forward pass"""
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)


class DetoxTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embedding_size: int,
        num_heads: int,
        vocab_size: int,
        feedforward_dim: int,
        max_size: int,
        dropout: float = 0.1,
    ):
        super(DetoxTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(
            embedding_size, dropout=dropout, max_size=max_size
        )
        self.input_embeddings = TokenEmbedding(vocab_size, embedding_size)
        self.output_embeddings = TokenEmbedding(vocab_size, embedding_size)
        self.transformer = Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )
        self.generator = nn.Linear(embedding_size, vocab_size)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
        src_padding_mask: Tensor,
        trg_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        """Make forward pass"""

        src_embeddings = self.positional_encoding(self.input_embeddings(src))
        trg_embeddings = self.positional_encoding(self.output_embeddings(trg))
        outs = self.transformer(
            src_embeddings,
            trg_embeddings,
            src_mask,
            trg_mask,
            None,
            src_padding_mask,
            trg_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        """Encode data using learned embeddings"""
        return self.transformer.encoder(
            self.positional_encoding(self.input_embeddings(src)), src_mask
        )

    def decode(self, trg: Tensor, memory: Tensor, trg_mask: Tensor):
        """Decode data using learned embeddings"""
        return self.transformer.decoder(
            self.positional_encoding(self.output_embeddings(trg)), memory, trg_mask
        )


### Trainer Classes


class CustomTransformerTrainer:
    """Custom Transformer model for Text Detoxification"""

    class DetoxificationDataset(torch.utils.data.Dataset):
        """Dataset for CustomTransformerTrainer"""

        def __init__(
            self,
            df: pd.DataFrame,
            tokenizer,
            special_symbols: list[str],
            unk_idx: int,
            bos_idx: int,
            eos_id: int,
        ):
            self.df = df
            self.tokenizer = tokenizer

            self.unk_idx, self.bos_idx, self.eos_idx = unk_idx, bos_idx, eos_id
            self.special_symbols = special_symbols

            self._preprocess()
            self._create_vocab()

        def _preprocess(self):
            # Clean columns
            self.df["toxic"] = self.df["toxic"].str.lower()
            self.df["nontoxic"] = self.df["nontoxic"].str.lower()

            # Tokenize sentences
            self.toxic = self.df["toxic"].apply(self.tokenizer).to_list()
            self.nontoxic = self.df["nontoxic"].apply(self.tokenizer).to_list()

            self.data = self.toxic + self.nontoxic

        def _create_vocab(self):
            self.vocab = build_vocab_from_iterator(
                self.data,
                min_freq=1,
                specials=self.special_symbols,
                special_first=True,
            )
            self.vocab.set_default_index(self.unk_idx)

        def _get_toxic(self, index: int) -> list:
            text = self.toxic[index]
            return [self.bos_idx, *self.vocab(text), self.eos_idx]

        def _get_nontoxic(self, index: int) -> list:
            text = self.nontoxic[index]
            return [self.bos_idx, *self.vocab(text), self.eos_idx]

        def __getitem__(self, index) -> tuple[list, list]:
            return self._get_toxic(index), self._get_nontoxic(index)

        def __len__(self) -> int:
            return len(self.toxic)

    def _collate_batch(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        toxic_batch, nontoxic_batch = [], []
        for _toxic, _nontoxic in batch:
            _toxic_tensor = torch.Tensor(_toxic)
            _nontoxic_tensor = torch.Tensor(_nontoxic)

            toxic_batch.append(_toxic_tensor[: self.max_size])
            nontoxic_batch.append(_nontoxic_tensor[: self.max_size])

        toxic_batch = pad_sequence(toxic_batch, padding_value=self.pad_idx)
        nontoxic_batch = pad_sequence(nontoxic_batch, padding_value=self.pad_idx)

        return toxic_batch.long(), nontoxic_batch.long()

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones((size, size), device=self.device)) == 1).transpose(
            0, 1
        )
        return (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

    def _create_mask(self, toxic, nontoxic):
        toxic_len = toxic.shape[0]
        nontoxic_len = nontoxic.shape[0]

        trg_mask = self._generate_square_subsequent_mask(nontoxic_len)
        src_mask = torch.zeros((toxic_len, toxic_len), device=self.device).type(
            torch.bool
        )

        trg_padding_mask = (nontoxic == self.pad_idx).transpose(0, 1)
        src_padding_mask = (toxic == self.pad_idx).transpose(0, 1)
        return src_mask, trg_mask, src_padding_mask, trg_padding_mask

    def _train_one_epoch(
        self,
        model,
        loader,
        optimizer,
        loss_fn,
        epoch,
    ):
        model.train()
        train_loss = 0.0
        total = 0

        if self.logger.verbose:
            loop = tqdm(
                loader,
                total=len(loader),
                desc=f"Epoch {epoch}: train",
                leave=True,
            )
        else:
            loop = loader

        for batch in loop:
            toxic, nontoxic = batch
            toxic, nontoxic = toxic.to(self.device), nontoxic.to(self.device)

            nontoxic_input = nontoxic[:-1, :]

            src_mask, trg_mask, src_padding_mask, trg_padding_mask = self._create_mask(
                toxic, nontoxic_input
            )

            # forward pass and loss calculation
            outputs = model(
                toxic,
                nontoxic_input,
                src_mask,
                trg_mask,
                src_padding_mask,
                trg_padding_mask,
                src_padding_mask,
            )

            nontoxic_out = nontoxic[1:, :]

            # zero the parameter gradients
            optimizer.zero_grad()

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), nontoxic_out.reshape(-1))

            # backward pass
            loss.backward()
            total += nontoxic.size(0)

            # optimizer run
            optimizer.step()

            train_loss += loss.item()
            if self.logger.verbose:
                loop.set_postfix({"loss": train_loss / total})

    def _val_one_epoch(
        self,
        model,
        loader,
        loss_fn,
        epoch,
    ) -> float:
        if self.logger.verbose:
            loop = tqdm(
                loader,
                total=len(loader),
                desc=f"Epoch {epoch}: val",
                leave=True,
            )
        else:
            loop = loader

        val_loss = 0.0
        total = 0
        with torch.no_grad():
            model.eval()  # evaluation mode
            for batch in loop:
                total += 1
                toxic, nontoxic = batch

                toxic, nontoxic = toxic.to(self.device), nontoxic.to(self.device)

                nontoxic_input = nontoxic[:-1, :]

                (
                    src_mask,
                    trg_mask,
                    src_padding_mask,
                    trg_padding_mask,
                ) = self._create_mask(toxic, nontoxic_input)

                outputs = model(
                    toxic,
                    nontoxic_input,
                    src_mask,
                    trg_mask,
                    src_padding_mask,
                    trg_padding_mask,
                    src_padding_mask,
                )

                nontoxic_out = nontoxic[1:, :]

                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]), nontoxic_out.reshape(-1)
                )

                val_loss += loss.item()
                if self.logger.verbose:
                    loop.set_postfix({"loss": val_loss / total})
        return val_loss / total

    def __init__(
        self,
        train_df: pd.DataFrame,
        cuda: bool,
        epochs: int,
        save_path: str,
        random_seed: int,
        logger: Logger,
    ) -> None:
        self.train_df = train_df
        self.cuda = cuda
        self.epochs = epochs
        self.save_path = save_path
        self.random_seed = random_seed
        self.logger = logger

        # Constants for the training process
        self.unk_idx, self.pad_idx, self.bos_idx, self.eos_idx = 0, 1, 2, 3
        self.special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.batch_size = 128
        self.max_size = 100
        self.device = torch.device(
            "cuda" if self.cuda and torch.cuda.is_available() else "cpu"
        )

    def train_and_save(self) -> None:
        """Train and save Custom Transformer model"""

        self.logger.log("Loading tokenizer...")
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

        self.logger.log("Creating dataset...")
        dataset = self.DetoxificationDataset(
            self.train_df,
            self.tokenizer,
            self.special_symbols,
            self.unk_idx,
            self.bos_idx,
            self.eos_idx,
        )

        self.logger.log("Splitting data...")
        train_dataset, val_dataset = random_split(
            dataset,
            [0.95, 0.05],
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        self.logger.log(f"{len(train_dataset)=}")
        self.logger.log(f"{len(val_dataset)=}")

        self.logger.log("Creating loaders...")
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_batch,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
        )

        self.logger.log("Creating model...")
        torch.manual_seed(self.random_seed)
        vocab_size = len(dataset.vocab)

        model = DetoxTransformer(4, 4, 320, 8, vocab_size, 512, self.max_size)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        model = model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        self.logger.log("Training...")
        best_loss = 1e10

        best = copy.deepcopy(model)
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch)
            val_loss = self._val_one_epoch(model, val_dataloader, loss_fn, epoch)
            if val_loss <= best_loss:
                val_loss = best_loss

        self.logger.log("Saving vocab...")
        torch.save(
            dataset.vocab, os.path.join(self.save_path, "custom_transformer_vocab.pth")
        )

        self.logger.log("Saving model...")
        torch.save(best, os.path.join(self.save_path, "custom_transformer"))


class BartTrainer:
    """Trainer class for fine-tunning BART for Text Detoxification task"""

    def _preprocess_function(self, data):
        inputs = [self.prefix + data_point for data_point in data["toxic"]]
        targets = data["nontoxic"]
        return self.tokenizer(
            inputs, text_target=targets, max_length=self.max_length, truncation=True
        )

    def _post_process_text(self, predictions: list, targets: list) -> tuple[list, list]:
        predictions = [pred.strip() for pred in predictions]
        targets = [label.strip() for label in targets]
        return predictions, targets

    def _compute_metrics(self, batch) -> dict:
        predictions, targets = batch
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        targets = np.where(targets != -100, targets, self.tokenizer.pad_token_id)  # type: ignore
        decoded_targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True)

        decoded_predictions, decoded_targets = self._post_process_text(
            decoded_predictions, decoded_targets
        )

        result = {}
        metrics = self.bleu_metric.compute(
            predictions=decoded_predictions, references=decoded_targets
        )
        if metrics is not None:
            result.update({"bleu": metrics["bleu"]})

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    def __init__(
        self,
        train_df: pd.DataFrame,
        cuda: bool,
        epochs: int,
        save_path: str,
        random_seed: int,
        logger: Logger,
    ) -> None:
        self.train_df = train_df
        self.cuda = cuda
        self.epochs = epochs
        self.save_path = save_path
        self.random_seed = random_seed
        self.logger = logger

        # Constants for the training process
        self.checkpoint: str = "eugenesiow/bart-paraphrase"
        self.batch_size = 16
        self.prefix = "paraphrase following to be nontoxic: \n"
        self.max_length = 128
        self.bleu_metric = evaluate.load("bleu")

    def train_and_save(self) -> None:
        """Train and save Bart model"""
        self.logger.log("Splitting data...")
        train_seq, val_seq = random_split(
            range(len(self.train_df)),  # type: ignore
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        train_indices, val_indices = (
            list(train_seq.indices),
            list(val_seq.indices),
        )

        self.logger.log(f"Loading model...\nCheckpoint: {self.checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.checkpoint
        )

        self.logger.log("Building datasets...")
        train_dataset = Dataset.from_pandas(self.train_df.iloc[train_indices]).map(
            self._preprocess_function, batched=True
        )
        val_dataset = Dataset.from_pandas(self.train_df.iloc[val_indices]).map(
            self._preprocess_function, batched=True
        )

        self.logger.log("Building trainer...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(self.save_path, "train_data", "bart"),
            evaluation_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            report_to=None,
            use_cpu=not self.cuda,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,  # type: ignore
            eval_dataset=val_dataset,  # type: ignore
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        self.logger.log("Training...")
        if self.logger.verbose:
            trainer.remove_callback(PrinterCallback)
        trainer.train()

        self.logger.log(f"Saving model...\nPath: {self.save_path}")
        trainer.save_model(os.path.join(self.save_path, "bart"))


def construct_absolute_path(*relative_path: str) -> str:
    """Turn relative file path to absolute

    Raises:
        FileNotFoundError

    Returns:
        str: absolute path
    """
    absolute_path = os.path.abspath(os.path.join(*relative_path))
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Path {absolute_path} does not exist")
    return absolute_path


def load_train_data(train_path: str, logger: Logger) -> pd.DataFrame:
    """Load train data from disk

    Args:
        train_path (str): path of training dataset
        logger (Logger): logger instance

    Returns:
        pd.DataFrame: training data pandas data frame
    """
    train_df = pd.read_csv(train_path)
    logger.log(f"{len(train_df)=}")

    return train_df


def train_model():
    """Train and save model"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and save model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model",
        choices=AVAILABLE_MODELS,
        default="bart",
        help="model to train (default: bart)",
    )
    parser.add_argument(
        "-d",
        "--dataset-size",
        type=str,
        choices=AVAILABLE_DATASET_SIZES,
        dest="dataset_size",
        default="xs",
        help="size of dataset used for train (default: xs)",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default="./models",
        help="relative path to save trained model (default: ./models)",
    )
    parser.add_argument(
        "-l",
        "--load-path",
        type=str,
        dest="load_path",
        default="./data/raw",
        help="relative path to load data from (default: ./data/raw)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        dest="epochs",
        default=20,
        help="how many epochs train model (default: 20)",
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        type=int,
        dest="random_seed",
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="use CUDA if available (default: True)",
    )
    parser.add_argument(
        "-i",
        "--ignore-warnings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="ignore warnings (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )

    namespace = parser.parse_args()
    (
        model,
        dataset_size,
        save_path,
        load_path,
        epochs,
        random_seed,
        cuda,
        ignore_warnings,
        verbose,
    ) = (
        namespace.model,
        namespace.dataset_size,
        namespace.save_path,
        namespace.load_path,
        namespace.epochs,
        namespace.random_seed,
        namespace.cuda,
        namespace.ignore_warnings,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)
    cuda: bool = bool(cuda)
    ignore_warnings: bool = bool(ignore_warnings)

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    save_path = construct_absolute_path(save_path)

    # Set up logger
    logger = Logger(verbose)

    # Load datasets
    load_train_path = construct_absolute_path(load_path, f"dataset_{dataset_size}.csv")

    train_df = load_train_data(load_train_path, logger)

    # Train and save model
    if model == "bart":
        trainer = BartTrainer(train_df, cuda, epochs, save_path, random_seed, logger)
    else:  # elif model == "custom_transformer"
        trainer = CustomTransformerTrainer(
            train_df, cuda, epochs, save_path, random_seed, logger
        )

    trainer.train_and_save()

    logger.log("Done!")


if __name__ == "__main__":
    train_model()
