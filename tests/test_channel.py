import numpy as np
import torch

from bench import chain
from channel import awgn, pad_channels, pad_to_multiple, power_normalize


def test_power_normalize_is_per_sample():
    symbols = torch.randn(4, 8, 6, 6) * torch.tensor([0.1, 1.0, 5.0, 20.0]).reshape(4, 1, 1, 1)
    normalized = power_normalize(symbols)
    complex_power = normalized.pow(2).sum(dim=(1, 2, 3)) / (8 * 6 * 6 / 2)
    assert torch.allclose(complex_power, torch.ones(4), atol=1e-5)


def test_awgn_matches_requested_snr_per_sample():
    torch.manual_seed(0)
    symbols = power_normalize(torch.randn(2, 64, 64, 64))
    snr_db = torch.tensor([[0.0], [15.0]])
    noise = awgn(symbols, snr_db) - symbols
    noise_power = noise.pow(2).sum(dim=(1, 2, 3)) / (symbols[0].numel() / 2)
    measured = 10 * torch.log10(1 / noise_power)
    assert torch.allclose(measured, snr_db.squeeze(1), atol=0.1)


def test_awgn_generator_is_reproducible():
    symbols = torch.zeros(2, 4, 3, 3)
    first = awgn(symbols, 10.0, generator=torch.Generator().manual_seed(3))
    second = awgn(symbols, 10.0, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_padding_helpers():
    assert pad_channels(torch.ones(1, 6, 2, 2), 10)[:, 6:].abs().sum() == 0
    assert pad_channels(torch.ones(1, 6, 2, 2), 10).shape == (1, 10, 2, 2)
    assert pad_to_multiple(torch.rand(1, 3, 50, 77), 8).shape == (1, 3, 56, 80)
    image = torch.rand(1, 3, 64, 64)
    assert pad_to_multiple(image, 8) is image


def test_numpy_chain_matches_torch():
    torch.manual_seed(0)
    symbols = torch.randn(3, 16, 4, 4)
    noise = torch.randn(3, 16, 4, 4)
    snr_db, active = 7.0, 10
    sent = power_normalize(symbols[:, :active])
    expected = pad_channels(sent + noise[:, :active] * torch.sqrt(torch.tensor(1 / (2 * 10 ** (snr_db / 10)))), 16)
    actual = chain.transmit(symbols.numpy(), noise.numpy(), snr_db, active, 16)
    assert np.allclose(actual, expected.numpy(), atol=1e-5)
    per_sample = chain.transmit(symbols.numpy(), noise.numpy(), np.full(3, snr_db), active, 16)
    assert np.allclose(per_sample, actual, atol=1e-6)
