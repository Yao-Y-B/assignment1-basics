from math import sqrt

import torch
from einops import einsum
from sympy import floor


class MyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = 2 / (in_features + out_features) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a = -3*std, b = 3*std)
        # self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight: (out, in)
        # x: in
        return einsum(x, self.weight,"batch sequence d_in, d_out d_in -> batch sequence d_out")

class MyEmbedding(torch.nn.Module):
    # the (batched) sequence of token IDs --> a sequence of vectors
    def __init__(self, num_embeddings : int, embedding_dim : int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        d_model = self.embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, d_model, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class MyRMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMSNorm logic

        rms_a = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        result = x / rms_a * self.weight

        return result.to(in_dtype)

class MySwiglu(torch.nn.Module):
    def __init__(self, d_model : int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        # d_ff = (8 * d_model) / 3
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(torch.empty(d_ff,d_model, device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty(d_model,d_ff, device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty(d_ff,d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(self.w1, x, "d_ff d_m, ... d_m -> ... d_ff")
        w3_x = einsum(self.w3, x, "d_ff d_m, ... d_m -> ... d_ff")
        silu_w1x = w1_x * torch.sigmoid(w1_x)
        hidden = silu_w1x * w3_x
        ffn_x = einsum(self.w2, hidden, "d_m d_ff, ... d_ff  -> ... d_m")
        return ffn_x

class MyRoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k : int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # self.max_seq_len = max_seq_len
        positions = torch.arange(max_seq_len).float()  # 位置编码 i
        # angle: theta(i,k) = i * (1/theta ** ((2k-2)/d) for k in {1, ..., d/2} 2(k-1) {0, 2, ... }
        angles = einsum(positions, 1.0 / self.theta ** ((2 * torch.arange(1, self.d_k // 2 + 1, 1) - 2) / self.d_k), "i,j->i j")

        self.register_buffer("sin", angles.sin(), persistent=False)
        self.register_buffer("cos", angles.cos(), persistent=False)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2])

        D = x.shape[-1]
        pair_dim = 2*floor(D // 2)

        # D 为奇数需要保留最后一维
        x_last = x[..., pair_dim:]

        # 可旋转部分
        x = x[..., :pair_dim]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        half_pair_dim = pair_dim // 2

        # sin cos维度保持一致
        sin = self.sin[token_positions, : half_pair_dim ]
        cos = self.cos[token_positions, : half_pair_dim ]

        x_t_even = x_even * cos - x_odd * sin
        x_t_odd = x_even * sin + x_odd * cos

        # 合并
        result = torch.stack([x_t_even, x_t_odd], dim=-1)
        result = result.flatten(-2)
        if pair_dim != D:
            result = torch.cat([result, x_last], dim=-1)

        return result