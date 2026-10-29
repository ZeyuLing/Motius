"""SentenceT5 tokenization and encode batching, outside the neural Bundle."""

from pathlib import Path
import json
import numpy as np
import torch


class SentenceT5Frontend:
    def __init__(self, network, directory):
        from transformers import AutoTokenizer

        self.network = network
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.options = json.loads(
            (Path(directory) / "sentence_bert_config.json").read_text()
        )

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        single = isinstance(texts, str)
        sentences = [texts] if single else list(texts)
        order = np.argsort([-len(sentence) for sentence in sentences])
        ordered = [sentences[index] for index in order]
        embeddings = []
        self.network.eval()
        device = next(self.network.parameters()).device
        for start in range(0, len(ordered), batch_size):
            batch = [
                str(sentence).strip()
                for sentence in ordered[start : start + batch_size]
            ]
            if self.options.get("do_lower_case", False):
                batch = [sentence.lower() for sentence in batch]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.options["max_seq_length"],
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            values = self.network(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
            )
            embeddings.extend(values.detach().cpu())
        embeddings = [embeddings[index] for index in np.argsort(order)]
        result = np.asarray([value.numpy() for value in embeddings])
        return result[0] if single else result
