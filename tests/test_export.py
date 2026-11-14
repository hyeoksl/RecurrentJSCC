import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from conftest import build, randomize_film

from export_onnx import TOLERANCE, check_static_graph, export_module


def trained_like(**overrides):
    model = build(**overrides)
    if overrides.get("film", "relative") != "none":
        randomize_film(model)
    return model.eval()


@pytest.mark.parametrize("role", ["encoder", "decoder"])
@pytest.mark.parametrize("iters", [1, 2, 4, 6])
@pytest.mark.parametrize("static_hw", [None, (64, 64)])
def test_onnx_matches_pytorch(tmp_path, role, iters, static_hw):
    model = trained_like()
    info = export_module(model[role], role, iters, tmp_path / "m.onnx", static_hw)
    assert info["max_abs_diff_vs_torch"] <= TOLERANCE
    onnx.checker.check_model(str(tmp_path / "m.onnx"))


@pytest.mark.parametrize("norm,act,grn", [("groupnorm1", "relu", True), ("batchnorm", "hardswish", False)])
@pytest.mark.parametrize("film", ["absolute", "none"])
def test_static_export_variants(tmp_path, norm, act, grn, film):
    model = trained_like(norm=norm, act=act, grn=grn, film=film)
    for role in ("encoder", "decoder"):
        info = export_module(model[role], role, 6, tmp_path / f"{role}.onnx", (64, 64))
        assert info["max_abs_diff_vs_torch"] <= TOLERANCE


def test_static_graph_has_only_fixed_float_inputs(tmp_path):
    model = trained_like()
    export_module(model["encoder"], "encoder", 2, tmp_path / "enc.onnx", (64, 96))
    export_module(model["decoder"], "decoder", 2, tmp_path / "dec.onnx", (64, 96))
    for name, expected in (("enc", {"image": [1, 3, 64, 96]}), ("dec", {"received": [1, 64, 8, 12]})):
        check_static_graph(tmp_path / f"{name}.onnx")
        session = ort.InferenceSession(str(tmp_path / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        inputs = {i.name: (i.shape, i.type) for i in session.get_inputs()}
        assert inputs == {**{k: (v, "tensor(float)") for k, v in expected.items()}, "snr": ([1, 1], "tensor(float)")}


def test_static_check_rejects_dynamic_graph(tmp_path):
    model = trained_like()
    export_module(model["encoder"], "encoder", 1, tmp_path / "dyn.onnx", None)
    with pytest.raises(ValueError):
        check_static_graph(tmp_path / "dyn.onnx")


def test_dynamic_export_accepts_other_sizes(tmp_path):
    model = trained_like()
    export_module(model["encoder"], "encoder", 2, tmp_path / "enc.onnx", None)
    session = ort.InferenceSession(str(tmp_path / "enc.onnx"), providers=["CPUExecutionProvider"])
    image, snr = torch.rand(3, 3, 40, 72), torch.tensor([[0.0], [10.0], [20.0]])
    with torch.no_grad():
        expected = model["encoder"](image, snr, 2).numpy()
    actual = session.run(None, {"image": image.numpy(), "snr": snr.numpy()})[0]
    assert np.abs(actual - expected).max() <= TOLERANCE


def test_film_is_folded_into_constants(tmp_path):
    model = trained_like()
    export_module(model["encoder"], "encoder", 4, tmp_path / "enc.onnx", (64, 64))
    ops = {node.op_type for node in onnx.load(str(tmp_path / "enc.onnx")).graph.node}
    assert not ops & {"Sin", "Cos", "Gemm", "MatMul"}
