import itertools

import pytest
import torch
from conftest import ITERS, build, randomize_film

from utils.complexity import count_flops, count_parameters, link_complexity
from utils.link import run_link


@pytest.mark.parametrize("enc_iters,dec_iters", list(itertools.product(ITERS, ITERS)))
def test_forward_backward_all_combinations(enc_iters, dec_iters, snr):
    model = build()
    images = torch.rand(2, 3, 64, 64)
    out = run_link(model, images, snr, 0.5, enc_iters, dec_iters)
    assert out["x_hat"].shape == images.shape
    assert out["y"].shape == (2, 64, 8, 8)
    out["x_hat"].mean().backward()
    for part in model.values():
        grads = [p.grad for p in part.core.steps.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("size", [(50, 77), (64, 71), (33, 64)])
def test_resolution_not_multiple_of_stride(size, snr):
    model = build()
    images = torch.rand(2, 3, *size)
    out = run_link(model, images, snr, 0.3, 2, 4)
    assert out["x_hat"].shape == images.shape
    assert out["y"].shape[1] == 2 * round(0.3 * 64)


@pytest.mark.parametrize("norm", ["layernorm2d", "groupnorm1", "batchnorm"])
@pytest.mark.parametrize("act,grn", [("gelu", False), ("relu", True), ("hardswish", False)])
def test_norm_act_grn_options(norm, act, grn, snr):
    model = build(norm=norm, act=act, grn=grn)
    out = run_link(model, torch.rand(2, 3, 64, 64), snr, 0.5, 6, 1)
    assert torch.isfinite(out["x_hat"]).all()


def test_parameter_count_independent_of_iters():
    counts = {count_parameters(build(iters=[k])) for k in ITERS} | {count_parameters(build())}
    assert len(counts) == 1
    info = [link_complexity(build(), 64, 64, k, k) for k in ITERS]
    assert len({(i["encoder_params"], i["decoder_params"]) for i in info}) == 1


def test_unshared_core_parameters_grow_with_depth():
    shared = build(film="none")
    core = count_parameters(shared["encoder"].core)
    for k in ITERS:
        unshared = build(film="none", share_core=False, iters=[k])
        assert count_parameters(unshared["encoder"].core) == k * core


def test_unshared_core_runs_only_its_depth(snr):
    model = build(film="none", share_core=False, iters=[2])
    images = torch.rand(2, 3, 64, 64)
    assert run_link(model, images, snr, 0.5, 2, 2)["x_hat"].shape == images.shape
    with pytest.raises(ValueError):
        run_link(model, images, snr, 0.5, 1, 2)
    with pytest.raises(ValueError):
        build(share_core=False)


def test_flops_difference_is_one_core_pass():
    model = build()
    for role in ("encoder", "decoder"):
        k1 = link_complexity(model, 64, 64, 1, 1)[f"{role}_gflops"]
        k2 = link_complexity(model, 64, 64, 2, 2)[f"{role}_gflops"]
        k6 = link_complexity(model, 64, 64, 6, 6)[f"{role}_gflops"]
        core = model[role].core
        latent = torch.zeros(1, 32, 8, 8)
        one_pass = count_flops(core, lambda core=core, latent=latent: core(latent, 1)) / 1e9
        assert one_pass > 0
        assert k2 - k1 == pytest.approx(one_pass, rel=1e-9)
        assert k6 - k1 == pytest.approx(5 * one_pass, rel=1e-9)


def test_film_modes_agree_at_zero_init(snr):
    reference = build(film="none")
    images = torch.rand(2, 3, 64, 64)
    outputs = []
    for mode in ("none", "relative", "absolute"):
        model = build(film=mode)
        missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
        assert not unexpected and all(".film." in key for key in missing)
        generator = torch.Generator().manual_seed(0)
        outputs.append(run_link(model.eval(), images, snr, 0.5, 4, 2, generator)["x_hat"])
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_relative_equals_absolute_at_k_max(snr):
    relative = build(film="relative")
    randomize_film(relative)
    absolute = build(film="absolute")
    absolute.load_state_dict(relative.state_dict())
    images = torch.rand(2, 3, 64, 64)

    def run(model, k):
        return run_link(model.eval(), images, snr, 0.5, k, k, torch.Generator().manual_seed(0))["x_hat"]

    k_max = max(ITERS)
    assert torch.equal(run(relative, k_max), run(absolute, k_max))
    assert not torch.allclose(run(relative, 2), run(absolute, 2))


def test_film_changes_output_once_trained(snr):
    model = build(film="relative")
    images = torch.rand(2, 3, 64, 64)
    before = run_link(model.eval(), images, snr, 0.5, 4, 4, torch.Generator().manual_seed(0))["x_hat"]
    randomize_film(model)
    after = run_link(model.eval(), images, snr, 0.5, 4, 4, torch.Generator().manual_seed(0))["x_hat"]
    assert not torch.allclose(before, after)


def test_held_out_depths_run(snr):
    for mode in ("relative", "absolute", "none"):
        model = build(film=mode)
        for k in (3, 5):
            assert torch.isfinite(run_link(model, torch.rand(2, 3, 64, 64), snr, 0.5, k, k)["x_hat"]).all()
