"""Vocal separators. Each takes a float32 stereo array (2, samples) at 44.1 kHz
and returns (vocals, instrumental) as arrays of the same shape.

  DemucsSeparator  Meta's HTDemucs (PyTorch). Splits into 4 stems; good quality,
                   heavy on CPU. Chunks are processed in batches.
  MDXSeparator     UVR MDX-Net vocal model (ONNX). Predicts vocals directly,
                   instrumental = mix - vocals. Runs on ONNX Runtime or on
                   Intel OpenVINO, which can also use an Intel Arc / iGPU.
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger("karaoke.separators")

SR = 44100


# --------------------------------------------------------------------------- Demucs
class DemucsSeparator:
    def __init__(self, name="htdemucs", device="cpu", overlap=0.1, batch=4, threads=0):
        import torch
        from demucs.pretrained import get_model

        if threads:
            torch.set_num_threads(threads)
        bag = get_model(name)
        # A "bag" may hold several models; htdemucs holds one. Use it directly
        # so we control chunking and batching ourselves.
        self.models = [m.to(device).eval() for m in getattr(bag, "models", [bag])]
        self.weights = getattr(bag, "weights", None)
        self.sources = bag.sources
        self.device, self.overlap, self.batch = device, overlap, max(1, batch)
        self.name = f"demucs:{name} overlap={overlap} batch={self.batch} on {device}"

    def separate(self, mix: np.ndarray, progress=None):
        import torch

        x = torch.from_numpy(mix)
        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        x = (x - mean) / std
        out = None
        for i, model in enumerate(self.models):
            y = self._run(model, x, progress)
            w = 1.0 if not self.weights else self.weights[i][self.sources.index("vocals")]
            out = y * w if out is None else out + y * w
        out = out * std + mean                                 # (sources, ch, n)
        vocals = out[self.sources.index("vocals")]
        instrumental = out.sum(0) - vocals
        return vocals.numpy(), instrumental.numpy()

    def _run(self, model, x, progress):
        import torch

        n = x.shape[-1]
        seg = int(float(model.segment) * SR)                   # 7.8 s for htdemucs
        stride = max(1, int(seg * (1 - self.overlap)))
        offsets = list(range(0, max(n - seg, 0) + stride, stride))
        # Triangle window: overlapping chunks fade into each other (as in demucs).
        half = torch.arange(1, seg // 2 + 1, dtype=torch.float32)
        win = torch.cat([half, half.flip(0)[: seg - len(half)]])
        win = win / win.max()
        out = torch.zeros(len(model.sources), x.shape[0], n)
        wsum = torch.zeros(n)
        for b in range(0, len(offsets), self.batch):
            offs = offsets[b:b + self.batch]
            chunks = []
            for o in offs:
                c = x[:, o:o + seg]
                if c.shape[-1] < seg:
                    c = torch.nn.functional.pad(c, (0, seg - c.shape[-1]))
                chunks.append(c)
            with torch.no_grad():
                y = model(torch.stack(chunks).to(self.device)).cpu()   # (B, sources, ch, seg)
            for k, o in enumerate(offs):
                L = min(seg, n - o)
                out[..., o:o + L] += y[k, ..., :L] * win[:L]
                wsum[o:o + L] += win[:L]
            if progress:
                progress(min((b + len(offs)) / len(offsets), 1.0))
        return out / wsum.clamp_min(1e-8)


# --------------------------------------------------------------------------- MDX-Net
MDX_MODELS = {
    # name: (file, n_fft, dim_f, dim_t, compensate)   -- values from UVR's model_data
    "Kim_Vocal_2": ("Kim_Vocal_2.onnx", 7680, 3072, 256, 1.009),
    "UVR-MDX-NET-Voc_FT": ("UVR-MDX-NET-Voc_FT.onnx", 7680, 3072, 256, 1.021),
}
MDX_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"


def ensure_mdx_model(name: str, models_dir: Path) -> Path:
    file = MDX_MODELS[name][0]
    path = models_dir / file
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s ...", file)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MDX_URL + file, tmp)
        tmp.replace(path)
    return path


class MDXSeparator:
    def __init__(self, name="Kim_Vocal_2", models_dir: Path = Path("models"), backend="openvino",
                 device="CPU", batch=1, threads=0):
        import torch

        self.file, self.n_fft, self.dim_f, self.dim_t, self.compensate = MDX_MODELS[name]
        path = ensure_mdx_model(name, models_dir)
        self.hop = 1024
        self.n_bins = self.n_fft // 2 + 1
        self.chunk = self.hop * (self.dim_t - 1)
        self.trim = self.n_fft // 2
        self.gen = self.chunk - 2 * self.trim
        self.window = torch.hann_window(self.n_fft, periodic=True)
        self.batch = max(1, batch)
        self._lock = threading.Lock()
        self.backend = backend
        if backend == "openvino":
            import openvino as ov
            core = ov.Core()
            # Cache compiled models: GPU compilation is slow the first time only.
            core.set_property({"CACHE_DIR": str(models_dir / "openvino_cache")})
            cfg = {"PERFORMANCE_HINT": "LATENCY"}
            if threads and device == "CPU":
                cfg["INFERENCE_NUM_THREADS"] = threads
            if device != "CPU":
                cfg["INFERENCE_PRECISION_HINT"] = "f16"     # GPUs run fp16 much faster
            m = core.read_model(str(path))
            m.reshape({m.inputs[0].get_any_name(): [self.batch, 4, self.dim_f, self.dim_t]})
            self._compiled = core.compile_model(m, device, cfg)
            self._infer = self._compiled.create_infer_request()
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if device == "DML" else ["CPUExecutionProvider"]
            self._sess = ort.InferenceSession(str(path), so, providers=providers)
        self.name = f"mdx:{name} {backend}/{device} batch={self.batch}"

    def _model(self, spec: np.ndarray) -> np.ndarray:
        if self.backend == "openvino":
            with self._lock:   # one infer request shared by threads
                n = spec.shape[0]
                if n < self.batch:
                    spec = np.concatenate([spec, np.zeros((self.batch - n, *spec.shape[1:]), np.float32)])
                return self._infer.infer({0: spec})[0][:n].copy()
        return self._sess.run(None, {"input": spec})[0]

    def _stft(self, x):
        import torch
        x = x.reshape(-1, self.chunk)
        s = torch.view_as_real(torch.stft(x, self.n_fft, self.hop, window=self.window, center=True, return_complex=True))
        s = s.permute(0, 3, 1, 2).reshape(-1, 2, 2, self.n_bins, self.dim_t).reshape(-1, 4, self.n_bins, self.dim_t)
        return s[:, :, : self.dim_f]

    def _istft(self, s):
        import torch
        pad = torch.zeros(s.shape[0], 4, self.n_bins - self.dim_f, self.dim_t)
        s = torch.cat([s, pad], -2).reshape(-1, 2, 2, self.n_bins, self.dim_t).reshape(-1, 2, self.n_bins, self.dim_t)
        s = torch.view_as_complex(s.permute(0, 2, 3, 1).contiguous())
        return torch.istft(s, self.n_fft, self.hop, window=self.window, center=True).reshape(-1, 2, self.chunk)

    def separate(self, mix: np.ndarray, progress=None):
        import torch
        n = mix.shape[-1]
        pad = self.gen - n % self.gen
        x = torch.from_numpy(np.concatenate(
            [np.zeros((2, self.trim), np.float32), mix, np.zeros((2, pad + self.trim), np.float32)], 1))
        offs = list(range(0, n + pad, self.gen))
        parts = []
        for b in range(0, len(offs), self.batch):
            chunk = torch.stack([x[:, o:o + self.chunk] for o in offs[b:b + self.batch]])
            spec = self._stft(chunk).numpy()
            y = self._istft(torch.from_numpy(self._model(spec)))
            parts.append(y[:, :, self.trim:-self.trim])
            if progress:
                progress(min((b + self.batch) / len(offs), 1.0))
        vocals = torch.cat(list(torch.cat(parts)), -1)[:, :n].numpy() * self.compensate
        return vocals.astype(np.float32), (mix - vocals).astype(np.float32)
