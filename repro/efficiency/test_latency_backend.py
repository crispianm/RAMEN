"""P2 unit test (no GPU): the timed post-stage hands the DEVICE tensor to the filter for every GPU backend, the numpy reference row still gets a host numpy array (D2H as released)."""
import os, sys, inspect
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pytest, torch
import latency_copy as LC
from evaluation import depth_probe as DP

class Rec:
    def __init__(self): self.got = []
    def __call__(self, x, guides): self.got.append(x)

def run(monkeypatch, backend, tokens=("agf0.2t0.06",)):
    monkeypatch.setattr(DP, "AGF_BACKEND", backend, raising=False)
    real = torch.rand
    monkeypatch.setattr(LC.torch, "rand", lambda *a, device=None, **k: real(*a))   # CPU stand-in for the 'device' probe (no GPU)
    rec = Rec(); p = {"depth_post": rec, "depth_post_tokens": list(tokens)}
    (name, stage), = LC.depth_post_stages_nosync(p, 8, "cuda")
    y = torch.zeros(1); assert stage(y, 0, 0) is y
    return rec.got[0]

@pytest.mark.parametrize("backend", ["gpu", "fast_triton_v3"])
def test_gpu_backends_receive_tensor(monkeypatch, backend):
    got = run(monkeypatch, backend)
    assert isinstance(got, torch.Tensor) and not isinstance(got, np.ndarray)      # device branch: no .cpu().numpy()

def test_numpy_reference_row_keeps_d2h(monkeypatch):
    got = run(monkeypatch, "cpu")                                                   # released numpy backend
    assert isinstance(got, np.ndarray)

def test_non_agf_tokens_stay_host(monkeypatch):
    assert isinstance(run(monkeypatch, "fast_triton_v3", ("ema",)), np.ndarray)

def test_cpu_device_never_device_branch():
    assert not LC.use_device_branch("cpu", "fast_triton_v3", ["agf0.2t0.06"])

def test_no_prestage_sync_and_end_event_after_last_op():
    assert "synchronize(" not in inspect.getsource(LC.depth_post_stages_nosync)
    src = inspect.getsource(LC.bj.cmd_latency)
    assert src.index("ev[k + 1].record()") < src.index("torch.cuda.synchronize()", src.index("ev[k + 1].record()")) < src.index("elapsed_time")

def test_precision_record_fields():
    assert set(LC.precision_record()) >= {"cudnn_allow_tf32", "matmul_allow_tf32", "float32_matmul_precision"}
