from typing import Dict, Optional, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (Dropout, Embedding, LayerNorm, Linear, Module,
                      ModuleList, Sequential)
from torch.nn.modules.activation import GELU

from tokenizer import Tokens


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

    def generate(self, x: Tensor, max_tokens: int = 100, method=Literal["sample", "greedy"], temperature=1.0) -> Tensor:
        # Note: need to implement temperature for GRPO
        generated_tokens = torch.IntTensor().to(self.device)
        original_size = x.shape[0]
        with torch.no_grad():
            for i in range(max_tokens):
                all_tokens = torch.concat([x, generated_tokens])
                causal_mask = (
                    torch.ones((all_tokens.shape[0], all_tokens.shape[0])).tril().bool()
                )
                causal_mask[:original_size, :original_size] = True

                model_out = self.forward(
                    all_tokens.unsqueeze(0), causal_mask.unsqueeze(0).to(self.device)
                )

                if method == "greedy":
                    prediction = torch.argmax(model_out.squeeze()[-1])
                else:
                    logits = model_out.squeeze()[-1]
                    updated_logits = logits / temperature
                    token_probs = torch.softmax(updated_logits, dim=0)
                    prediction = torch.multinomial(token_probs, 1)[0]

                generated_tokens = torch.concat([generated_tokens, prediction.unsqueeze(0)])
                if prediction.item() == Tokens.TOKEN_EOS.value:
                    break

        return generated_tokens

    def generate_rollouts(self, x: Tensor, max_tokens: int = 100) -> Tensor:
        # Now that I think about it, I don't actually know too much about
        # how batched generation works efficiently here. There are multiple
        # sequences, all of different lengths, and we'll need to manage the
        # causal masks for all of them.
        pass
