from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Self

import torch
import torch.nn.functional as F
from torch import IntTensor, Tensor
from torch.nn import (Dropout, Embedding, LayerNorm, Linear, Module,
                      ModuleList, Sequential)
from torch.nn.modules.activation import GELU
from torch.nn.utils.rnn import pad_sequence

from tokenizer import Tokens
from training_utils import create_causal_mask


class MultiHeadAttention(Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0, device: str = "cuda"
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = dropout
        self.device = device

        self.qkv = Linear(d_model, 3 * d_model, device=device)
        self.out_proj = Linear(d_model, d_model, device=device)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        B, T, _ = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_scores = F.scaled_dot_product_attention(
            q, k, v, causal_mask, dropout_p=self.dropout if self.training else 0
        )
        attn_scores = attn_scores.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(attn_scores)


class TransformerBlock(Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0, device: str = "mps"
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, device)
        self.ln1 = LayerNorm(d_model, device=device)
        self.ln2 = LayerNorm(d_model, device=device)
        self.mlp = Sequential(
            Linear(d_model, d_model * 4, device=device),
            GELU(),
            Linear(d_model * 4, d_model, device=device),
            Dropout(dropout),
        )

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MazeTransformer(Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.output_vocab_size = config["output_vocab_size"]
        self.device = config["device"]
        self.d_model = config["d_model"]
        self.max_seq_len = config["max_seq_len"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]

        self.embedding_layer = Embedding(
            self.vocab_size, self.d_model, device=self.device
        )
        self.pos_embedding_layer = Embedding(
            self.max_seq_len, self.d_model, device=self.device
        )
        self.transformer_blocks = ModuleList(
            [
                TransformerBlock(
                    self.d_model, self.n_heads, self.dropout, device=self.device
                )
                for _ in range(self.n_layers)
            ]
        )
        self.ln = LayerNorm(self.d_model, device=self.device)
        self.head = Linear(
            self.d_model, self.output_vocab_size, device=self.device
        )  # The vocab size on this is only 5 -> up, down, left, right, and EOS

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        token_embeddings = self.embedding_layer(x)
        pos_embeddings = self.pos_embedding_layer(
            torch.arange(x.size(1), device=self.device)
        )
        x = token_embeddings + pos_embeddings

        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x, causal_mask)

        x = self.ln(x)
        logits = self.head(x)
        return logits

    def _sample_with_temperature(self, logits: Tensor, temperature: float):
        # logits is of shape (B, output_vocab_size)
        assert temperature >= 0, "Temperature must be greater than 0!"
        if temperature == 0:
            return torch.argmax(logits, dim=1)
        updated_logits = logits / temperature
        token_probs = torch.softmax(updated_logits, dim=0)
        return token_probs, torch.multinomial(token_probs, 1).squeeze()

    def generate(
        self,
        x: Tensor,
        method: Literal["sample", "greedy"] = "greedy",
        temperature=1.0,
    ) -> Tensor:
        # Note: need to implement temperature for GRPO
        generated_tokens = torch.IntTensor().to(self.device)
        original_size = x.shape[0]
        with torch.no_grad():
            for _ in range(self.max_seq_len):
                all_tokens = torch.concat([x, generated_tokens])
                causal_mask = (
                    torch.ones((all_tokens.shape[0], all_tokens.shape[0])).tril().bool()
                )
                causal_mask[:original_size, :original_size] = True

                model_out = self.forward(
                    all_tokens.unsqueeze(0), causal_mask.unsqueeze(0).to(self.device)
                )

                logits = model_out.squeeze()[-1]
                _, prediction = self._sample_with_temperature(
                    logits, 0 if method == "greedy" else temperature
                )

                generated_tokens = torch.concat(
                    [generated_tokens, prediction.unsqueeze(0)]
                )
                if prediction.item() == Tokens.TOKEN_EOS.value:
                    break

        return generated_tokens

    def generate_rollouts(
        self,
        x: Tensor,
        sizes: IntTensor,
        method: Literal["sample", "greedy"] = "sample",
        temperature: float = 1.0,
        baseline_model: "MazeTransformer" = None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        # This is definitely not optimized, but I don't want to spend too much
        # time working on the inference logic here.
        # Improvements:
        #  - We only run max_seq_len - sizes.max() iterations. This means that
        #    if there's a sequence that fits that needs more iterations it'll
        #    stop early.
        #  - KV cache
        #
        # I hate how the number of things this function returns is variable
        # depending on whether or not baseline_model is passed in. Can future
        # Yash please clean this up.

        predictions = [[] for _ in range(x.shape[0])]
        pred_token_probs = [torch.Tensor().to(self.device) for _ in range(x.shape[0])]
        ref_token_probs = [torch.Tensor().to(self.device) for _ in range(x.shape[0])]

        rollout_results = deepcopy(x).to(self.device)
        finished = torch.zeros((1, rollout_results.shape[0])).bool().to(self.device)
        for idx in range(self.max_seq_len - sizes.max()):
            batch_size, seq_len = rollout_results.shape
            causal_masks = create_causal_mask(sizes, seq_len).to(self.device)
            model_out = self.forward(
                rollout_results, causal_masks
            )

            if baseline_model:
                baseline_out = baseline_model.forward(rollout_results, causal_masks)
                baseline_logits = baseline_out[torch.arange(batch_size), sizes + idx - 1]
                baseline_token_probs, _ = self._sample_with_temperature(baseline_logits, temperature)

            logits = model_out[torch.arange(batch_size), sizes + idx - 1]
            token_probs, predicted = self._sample_with_temperature(
                logits, 0 if method == "greedy" else temperature
            )
            for i in range(predicted.shape[0]):
                if not finished[0, i].item():
                    prediction = predicted[i].item()
                    predictions[i].append(prediction)
                    pred_token_probs[i] = torch.cat([pred_token_probs[i], token_probs[i][prediction].unsqueeze(0)])

                    if baseline_model:
                        ref_token_probs[i] = torch.cat([ref_token_probs[i], baseline_token_probs[i][prediction].unsqueeze(0)])

            zeros = torch.zeros(batch_size, 1).to(self.device).long()
            rollout_results = torch.cat([rollout_results, zeros], dim=1)
            rollout_results[torch.arange(batch_size), sizes + idx] = predicted

            finished = torch.logical_or(finished, predicted == Tokens.TOKEN_EOS.value)
            if finished.sum().item() == batch_size:
                break

        predictions = [torch.IntTensor(rollout) for rollout in predictions]

        if baseline_model:
            return predictions, pred_token_probs, ref_token_probs

        return predictions, pred_token_probs
