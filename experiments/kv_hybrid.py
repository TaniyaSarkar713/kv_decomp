# kv_hybrid.py -- shared machinery for the hybrid old-cache experiments
# (stages 7+), distilled from stage6_shufflerless_phase0.py:
#   dense prefill -> compress ONLY the aged old block (hot window exact) ->
#   generate/evaluate with the compressed cache.
#
# Components: post-RoPE query capture, offline fixed-basis projection
# (stage-6 calibration file), observed-attention-mass computation, SAGE
# influence / energy RoPE-pair residual selection, KIVI-style old-block
# quantization, and true int4 packing (for physical-bytes stages).

import os
import torch
from transformers import DynamicCache

from kv_common import kivi_quant_dequant


class Hybrid:
    """Holds model geometry, q-capture hooks, and the stage-6 calibration."""

    def __init__(self, model, calib_path="results/stage6_calib.pt"):
        self.model = model
        cfg = model.config
        self.cfg = cfg
        self.L = cfg.num_hidden_layers
        self.HQ = cfg.num_attention_heads
        self.H = cfg.num_key_value_heads
        self.G = self.HQ // self.H
        self.D = cfg.head_dim
        self.P = self.D // 2
        self.dev = model.device
        assert os.path.exists(calib_path), (
            f"{calib_path} missing -- run experiments/stage6_shufflerless_phase0.py "
            "once to produce the offline calibration")
        c = torch.load(calib_path, weights_only=True)
        self.UFULL = c["UFULL"].to(self.dev)      # [L, 2, H, D, D]
        self.QBAR = c["QBAR"].to(self.dev)        # [L, H, P]
        self._cap = {"on": False, "store": []}

        def hook(module, inp, out):
            if self._cap["on"]:
                self._cap["store"].append(out.detach())

        for lyr in model.model.layers:
            lyr.self_attn.q_proj.register_forward_hook(hook)

    # ---------------- RoPE / query capture ----------------
    def _rotate_half(self, x):
        return torch.cat((-x[..., self.D // 2:], x[..., : self.D // 2]), dim=-1)

    def _rope_cos_sin(self, S):
        pos = torch.arange(S, device=self.dev).unsqueeze(0)
        dummy = torch.ones(1, S, 1, dtype=torch.bfloat16, device=self.dev)
        cos, sin = self.model.model.rotary_emb(dummy, pos)
        return cos.float(), sin.float()

    def _q_postrope(self, S):
        cos, sin = self._rope_cos_sin(S)
        out = []
        for q in self._cap["store"]:
            q = q[0].float().view(-1, self.HQ, self.D).transpose(0, 1)
            out.append(q * cos + self._rotate_half(q) * sin)
        return out

    def prefill(self, ids, capture_q=False):
        """Dense prefill of ids[:, :-1] (last token left for generate/eval).
        Returns (per-layer [(K, V)], q_list or None)."""
        self._cap["store"] = []
        self._cap["on"] = capture_q
        c = DynamicCache(config=self.cfg)
        with torch.no_grad():
            self.model(input_ids=ids[:, :-1], past_key_values=c, use_cache=True,
                       logits_to_keep=1)
        self._cap["on"] = False
        q_list = self._q_postrope(ids.shape[1] - 1) if capture_q else None
        self._cap["store"] = []
        return [(c[i][0], c[i][1]) for i in range(len(c))], q_list

    def attention_mass(self, base, q_list, W):
        """abar per layer [H, S_old]: mean attention mass of the last-W prefill
        queries on old tokens (grouped-query heads averaged)."""
        S_p = base[0][0].shape[-2]
        out = []
        qpos = S_p - W + torch.arange(W, device=self.dev)
        mask = torch.arange(S_p, device=self.dev)[None, :] > qpos[:, None]
        for li, (K, V) in enumerate(base):
            q = q_list[li][:, -W:]
            sc = torch.einsum("hqd,htd->hqt", q,
                              K[0].float().repeat_interleave(self.G, 0)) / self.D**0.5
            pr = sc.masked_fill(mask[None], -torch.inf).softmax(-1)
            out.append(pr.view(self.H, self.G, W, S_p).mean(dim=(1, 2))[:, : S_p - W])
        return out

    # ---------------- codecs for the old block ----------------
    def project_fixed(self, X, li, ti, r):
        U = self.UFULL[li, ti, :, :, :r]
        A = torch.einsum("hsd,hdr->hsr", X, U)
        return torch.einsum("hsr,hdr->hsd", A, U)

    def _pair_norm(self, E):
        return (E[..., : self.P] ** 2 + E[..., self.P:] ** 2 + 1e-20).sqrt()

    def _top_pairs(self, score, m):
        flat = score.reshape(-1)
        idx = torch.topk(flat, min(m, flat.numel())).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        return mask.view(score.shape)

    def residual(self, E_K, E_V, li, rule, rho, abar=None):
        """Sparse RoPE-pair residual for one layer. rule: influence|energy."""
        S_old = E_K.shape[1]
        m = int(rho * self.H * S_old * self.P)
        pnK, pnV = self._pair_norm(E_K), self._pair_norm(E_V)
        if rule == "influence":
            a = abar[li].unsqueeze(-1)
            mk = self._top_pairs(a * self.QBAR[li].unsqueeze(1) * pnK, m)
            mv = self._top_pairs(a * pnV, m)
        elif rule == "energy":
            mk = self._top_pairs(pnK, m)
            mv = self._top_pairs(pnV, m)
        else:
            raise ValueError(rule)
        keepK = torch.cat([mk, mk], dim=-1)
        keepV = torch.cat([mv, mv], dim=-1)
        return E_K * keepK, E_V * keepV

    def compress(self, base, spec, W, abar=None):
        """Per-layer (K, V) with the old block (S-W tokens) re-coded per `spec`:
          {"mode": "copy" | "latent" | "kivi", "r": int,
           "residual": None | (rule, rho)}
        Hot window (last W) and later decode tokens stay exact."""
        out = []
        for li, (K, V) in enumerate(base):
            S = K.shape[-2]
            if spec["mode"] == "copy":
                out.append((K, V))
                continue
            Kf = K[0, :, : S - W].float()
            Vf = V[0, :, : S - W].float()
            if spec["mode"] == "latent":
                r = spec["r"]
                Kh = self.project_fixed(Kf, li, 0, r)
                Vh = self.project_fixed(Vf, li, 1, r)
                if spec.get("residual"):
                    rule, rho = spec["residual"]
                    eK, eV = self.residual(Kf - Kh, Vf - Vh, li, rule, rho, abar)
                    Kh, Vh = Kh + eK, Vh + eV
            elif spec["mode"] == "kivi":
                Kq, Vq, _ = kivi_quant_dequant(
                    Kf.unsqueeze(0), Vf.unsqueeze(0),
                    group=spec.get("group", 32), residual=0,
                    bits=spec.get("bits", 2))
                Kh, Vh = Kq[0], Vq[0]
            else:
                raise ValueError(spec["mode"])
            Kc = torch.cat([Kh.unsqueeze(0).to(K.dtype), K[:, :, S - W:]], dim=-2)
            Vc = torch.cat([Vh.unsqueeze(0).to(V.dtype), V[:, :, S - W:]], dim=-2)
            out.append((Kc, Vc))
        return out

    def build_cache(self, tensors):
        c = DynamicCache(config=self.cfg)
        for i, (K, V) in enumerate(tensors):
            c.update(K, V, i)
        return c

    def old_block_bytes(self, spec, S_old):
        """Analytic physical bytes/token/layer for the old block (K+V, bf16
        coefficients, pair residual = 2 values bf16 + int32 index)."""
        dense = 2 * self.H * self.D * 2
        if spec["mode"] == "copy":
            return dense
        if spec["mode"] == "latent":
            b = 2 * self.H * spec["r"] * 2
            if spec.get("residual"):
                _, rho = spec["residual"]
                b += 2 * rho * self.H * self.P * (2 * 2 + 4)
            return b
        if spec["mode"] == "kivi":
            bits, group = spec.get("bits", 2), spec.get("group", 32)
            return 2 * self.H * self.D * (bits + 2 * 16.0 / group) / 8
        raise ValueError(spec["mode"])


# ---------------- true low-bit packing (physical storage, stage 8+) ----------------
def pack_quant(x, group, grp_dim, bits=4):
    """Asymmetric 2/4-bit quantization with REAL packed uint8 storage. x fp32;
    quantization groups of `group` along grp_dim (its size must divide by
    `group`; total numel divisible by 8//bits). Returns
    (packed uint8, mn fp16, scale fp16, physical_bytes, meta)."""
    assert bits in (2, 4)
    vpb = 8 // bits
    levels = (1 << bits) - 1
    xm = x.movedim(grp_dim, -1).contiguous()
    orig = xm.shape
    xs = xm.reshape(*orig[:-1], orig[-1] // group, group)
    mn = xs.amin(-1, keepdim=True)
    mx = xs.amax(-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / levels
    q = ((xs - mn) / scale).round().clamp(0, levels).to(torch.uint8).reshape(-1)
    packed = torch.zeros(q.numel() // vpb, dtype=torch.uint8, device=x.device)
    for i in range(vpb):
        packed |= q[i::vpb] << (bits * i)
    nbytes = packed.numel() + (mn.numel() + scale.numel()) * 2  # fp16 stats
    return packed, mn.half(), scale.half(), nbytes, (tuple(orig), grp_dim)


def unpack_quant(packed, mn, scale, meta, group, bits=4):
    vpb = 8 // bits
    mask = (1 << bits) - 1
    orig, grp_dim = meta
    parts = [(packed >> (bits * i)) & mask for i in range(vpb)]
    q = torch.stack(parts, -1).reshape(*orig[:-1], orig[-1] // group, group)
    x = (q.float() * scale.float() + mn.float()).reshape(orig)
    return x.movedim(-1, grp_dim)
