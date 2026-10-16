"""ClaimSet Transformer: learn directly from a provider's raw billing lines.

Each provider-year is a *set* of up to ``MAX_LINES`` (HCPCS code, place of service)
lines with per-line volume, price and peer-relative intensity. A transformer
encoder (no positional encoding: the set is unordered) pools the lines into a
provider embedding, which feeds two heads:

  * fraud head      -> P(excluded within window), given the embedding + specialty
  * specialty head  -> auxiliary multi-task target predicted from the code mix
                       alone. Every provider has a specialty label, so this gives
                       the encoder ~1M labels per year instead of ~300 positives.

The specialty is withheld from the encoder, so the auxiliary task cannot be solved
trivially, and the fraud head can learn "bills unlike its declared specialty".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn

MAX_LINES = 32
LINE_NUM = ["log_srv", "log_benes", "log_spb", "log_pay_unit", "lr_intensity", "lr_charge",
            "lr_pay", "pay_share", "srv_per_day"]
PAD, UNK, EMPTY = 0, 1, 2  # reserved code ids


@dataclass
class LineTensors:
    codes: np.ndarray   # [N, L] int32
    pos: np.ndarray     # [N, L] int8   (0 pad, 1 office, 2 facility)
    num: np.ndarray     # [N, L, F] float16
    mask: np.ndarray    # [N, L] bool, True = real line


def build_vocab(svcs: list[pl.DataFrame], min_providers: int = 20) -> dict[str, int]:
    counts = pl.concat([s.select("hcpcs", "npi") for s in svcs]).unique() \
        .group_by("hcpcs").len().filter(pl.col("len") >= min_providers)
    codes = sorted(counts["hcpcs"].to_list())
    return {c: i + 3 for i, c in enumerate(codes)}


def line_tensors(svc: pl.DataFrame, npis: list[str], vocab: dict[str, int]) -> LineTensors:
    """Top-``MAX_LINES`` lines by payment for each NPI in ``npis`` (in that order)."""
    s = svc.with_columns(
        (pl.col("Tot_Srvcs") * pl.col("Avg_Mdcr_Stdzd_Amt")).alias("pay"),
        (pl.col("Tot_Srvcs") / pl.col("Tot_Benes")).alias("spb"),
    )
    g = ["specialty", "hcpcs"]
    s = s.with_columns(
        (pl.col("spb") / pl.col("spb").median().over(g)).log().alias("lr_intensity"),
        (pl.col("Avg_Sbmtd_Chrg") / pl.col("Avg_Sbmtd_Chrg").median().over(g)).log()
        .alias("lr_charge"),
        (pl.col("Avg_Mdcr_Stdzd_Amt") / pl.col("Avg_Mdcr_Stdzd_Amt").median().over(g)).log()
        .alias("lr_pay"),
        (pl.col("pay") / pl.col("pay").sum().over("npi")).alias("pay_share"),
        (pl.col("Tot_Srvcs") / pl.col("Tot_Bene_Day_Srvcs")).log().alias("srv_per_day"),
        pl.col("Tot_Srvcs").log1p().alias("log_srv"),
        pl.col("Tot_Benes").log1p().alias("log_benes"),
        pl.col("spb").log().alias("log_spb"),
        pl.col("Avg_Mdcr_Stdzd_Amt").log1p().alias("log_pay_unit"),
        pl.col("hcpcs").replace_strict(vocab, default=UNK, return_dtype=pl.Int32).alias("code"),
        pl.when(pl.col("pos") == "F").then(2).otherwise(1).cast(pl.Int8).alias("pos_id"),
    )
    s = s.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0.0).clip(-10, 20).alias(c)
        for c in LINE_NUM
    ])
    order = pl.DataFrame({"npi": npis, "row": np.arange(len(npis))})
    s = (s.join(order, on="npi", how="inner")
         .sort(["row", "pay"], descending=[False, True])
         .with_columns(pl.int_range(pl.len()).over("row").alias("slot"))
         .filter(pl.col("slot") < MAX_LINES))

    n = len(npis)
    codes = np.zeros((n, MAX_LINES), np.int32)
    pos = np.zeros((n, MAX_LINES), np.int8)
    num = np.zeros((n, MAX_LINES, len(LINE_NUM)), np.float16)
    r, c = s["row"].to_numpy(), s["slot"].to_numpy()
    codes[r, c] = s["code"].to_numpy()
    pos[r, c] = s["pos_id"].to_numpy()
    num[r, c] = s.select(LINE_NUM).to_numpy().astype(np.float16)
    empty = ~(codes > 0).any(1)
    codes[empty, 0] = EMPTY  # providers without service lines get a single marker token
    pos[empty, 0] = 1
    return LineTensors(codes, pos, num, codes > 0)


class ClaimSetTransformer(nn.Module):
    def __init__(self, n_codes: int, n_specialties: int, n_aux: int, n_prov: int,
                 d: int = 96, heads: int = 4, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.code = nn.Embedding(n_codes, d, padding_idx=PAD)
        self.pos = nn.Embedding(3, d)
        self.num = nn.Sequential(nn.Linear(len(LINE_NUM), d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.attn_pool = nn.Linear(d, 1)
        self.spec = nn.Embedding(n_specialties, 16)
        self.aux_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_aux))
        self.fraud_head = nn.Sequential(
            nn.Linear(d + 16 + n_prov + 1, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def encode(self, codes, pos, num, mask):
        x = self.code(codes) + self.pos(pos.long()) + self.num(num)
        x = self.encoder(self.norm(x), src_key_padding_mask=~mask)
        w = self.attn_pool(x).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        h = (w.unsqueeze(-1) * x).sum(1)
        return h, self.aux_head(h)

    def forward(self, codes, pos, num, mask, spec, prov, aux_target=None):
        h, aux_logits = self.encode(codes, pos, num, mask)
        # Surprise of the declared specialty under the code-mix model (detached so the
        # fraud loss cannot distort the specialty classifier).
        if aux_target is not None:
            logp = aux_logits.detach().log_softmax(-1)
            surprise = -logp.gather(1, aux_target[:, None])
        else:
            surprise = torch.zeros(len(h), 1, device=h.device)
        z = torch.cat([h, self.spec(spec), prov, surprise], -1)
        return self.fraud_head(z).squeeze(-1), aux_logits


def _batches(idx: np.ndarray, bs: int):
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


class Trainer:
    def __init__(self, lt: LineTensors, spec: np.ndarray, aux: np.ndarray, prov: np.ndarray,
                 n_codes: int, n_specialties: int, n_aux: int, device: str | None = None,
                 seed: int = 0, log=print, **model_kw):
        self.lt, self.spec, self.aux, self.prov = lt, spec, aux, prov
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.model = ClaimSetTransformer(n_codes, n_specialties, n_aux, prov.shape[1],
                                         **model_kw).to(self.device)
        self.log = log

    def _tensors(self, b: np.ndarray):
        dev = self.device
        return (torch.from_numpy(self.lt.codes[b]).long().to(dev),
                torch.from_numpy(self.lt.pos[b]).to(dev),
                torch.from_numpy(self.lt.num[b].astype(np.float32)).to(dev),
                torch.from_numpy(self.lt.mask[b]).to(dev),
                torch.from_numpy(self.spec[b]).long().to(dev),
                torch.from_numpy(self.prov[b]).float().to(dev),
                torch.from_numpy(self.aux[b]).long().to(dev))

    @torch.no_grad()
    def predict(self, idx: np.ndarray, bs: int = 4096) -> np.ndarray:
        self.model.eval()
        out = []
        for b in _batches(idx, bs):
            c, p, n, m, s, pv, a = self._tensors(b)
            logit, _ = self.model(c, p, n, m, s, pv, a)
            out.append(logit.float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed(self, idx: np.ndarray, bs: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Pooled code-mix embedding and specialty surprise for each provider."""
        self.model.eval()
        hs, sur = [], []
        for b in _batches(idx, bs):
            c, p, n, m, s, pv, a = self._tensors(b)
            h, aux_logits = self.model.encode(c, p, n, m)
            hs.append(h.float().cpu().numpy())
            sur.append((-aux_logits.log_softmax(-1).gather(1, a[:, None])).squeeze(1)
                       .float().cpu().numpy())
        return np.concatenate(hs), np.concatenate(sur)

    def fit(self, train_idx: np.ndarray, y: np.ndarray, valid_idx: np.ndarray | None = None,
            y_valid: np.ndarray | None = None, epochs: int = 12, neg_per_epoch: int = 200_000,
            pos_repeat: int = 20, bs: int = 1024, lr: float = 1e-3, aux_weight: float = 0.3,
            fraud_weight: float = 1.0):
        pos_idx = train_idx[y[train_idx] == 1]
        neg_idx = train_idx[y[train_idx] == 0]
        steps_per_epoch = math.ceil((neg_per_epoch + pos_repeat * len(pos_idx)) / bs)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps_per_epoch,
                                                    pct_start=0.1)
        # Negatives are subsampled; reweight them so the loss matches the full data.
        neg_w = len(neg_idx) / neg_per_epoch / max(pos_repeat, 1)
        history, best_state, best_ap = [], None, -1.0
        for ep in range(epochs):
            t = time.time()
            self.model.train()
            ep_idx = np.concatenate([np.repeat(pos_idx, pos_repeat),
                                     self.rng.choice(neg_idx, neg_per_epoch, replace=False)])
            self.rng.shuffle(ep_idx)
            tot = 0.0
            for b in _batches(ep_idx, bs):
                c, p, n, m, s, pv, a = self._tensors(b)
                yb = torch.from_numpy(y[b]).float().to(self.device)
                logit, aux_logits = self.model(c, p, n, m, s, pv, a)
                w = torch.where(yb > 0, 1.0, neg_w)
                loss = (fraud_weight * F.binary_cross_entropy_with_logits(logit, yb, weight=w)
                        / (neg_w + 1) + aux_weight * F.cross_entropy(aux_logits, a))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            row = {"epoch": ep + 1, "loss": tot / steps_per_epoch}
            if valid_idx is not None and fraud_weight > 0:
                row["valid_ap"] = float(average_precision_score(y_valid, self.predict(valid_idx)))
                if row["valid_ap"] > best_ap:
                    best_ap = row["valid_ap"]
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            history.append(row)
            self.log(f"    epoch {ep + 1:2d} loss={row['loss']:.4f} "
                     f"valid_ap={row.get('valid_ap', float('nan')):.4f} [{time.time() - t:.0f}s]")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history
