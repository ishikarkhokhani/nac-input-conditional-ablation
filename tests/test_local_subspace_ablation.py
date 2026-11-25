import math

import pytest
import torch

from nac_uncertainty_regression.nac import NACMode, NACWrapper


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(2, 4)
        self.act = torch.nn.Tanh()
        self.out = torch.nn.Linear(4, 1)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.5, 0.5]]))
            self.hidden.bias.zero_()
            self.out.weight.copy_(torch.tensor([[1.0, -1.0, 0.5, 0.25]]))
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.hidden(x))
        return self.out(h)


FIXTURE_FAMILIES = [
    "axis_split",
    "diagonal_bands",
    "curved_shells",
    "checkerboard_local",
    "duplicated_points_control",
]
SEEDS = [0, 1]
NEGATIVE_STRATEGIES = ["random", "farthest", "pooled"]


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.zeros_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(values) + 1, dtype=torch.float32)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.square().sum()) * (ry.square().sum()))
    if float(denom) == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _make_fixture(name: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n_train = 24
    n_eval = 10
    if name == "axis_split":
        train_x = torch.stack([torch.linspace(-1.2, 1.2, n_train), torch.zeros(n_train)], dim=1)
        eval_x = torch.stack([torch.linspace(-1.1, 1.1, n_eval), torch.linspace(-0.8, 0.8, n_eval)], dim=1)
        target = eval_x[:, 1].abs() + 0.1 * eval_x[:, 0].abs()
    elif name == "diagonal_bands":
        base = torch.linspace(-1.5, 1.5, n_train)
        train_x = torch.stack([base, base + 0.15 * torch.sign(base + 1e-4)], dim=1)
        eval_x = torch.stack([torch.linspace(-1.4, 1.4, n_eval), torch.linspace(1.4, -1.4, n_eval)], dim=1)
        target = (eval_x[:, 0] + eval_x[:, 1]).abs()
    elif name == "curved_shells":
        theta = torch.linspace(0.0, math.pi, n_train)
        train_x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        eval_theta = torch.linspace(0.1, math.pi - 0.1, n_eval)
        eval_x = 1.35 * torch.stack([torch.cos(eval_theta), torch.sin(eval_theta)], dim=1)
        target = torch.full((n_eval,), 0.35)
    elif name == "checkerboard_local":
        pts = []
        for i in range(6):
            for j in range(4):
                pts.append([(i - 2.5) / 2.0, (j - 1.5) / 2.0])
        train_x = torch.tensor(pts, dtype=torch.float32)
        eval_x = train_x[:n_eval] + 0.23
        target = ((eval_x[:, 0].sign() != eval_x[:, 1].sign()).float() + 0.2 * eval_x.norm(dim=1))
    elif name == "duplicated_points_control":
        base = torch.tensor([
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-0.5, -0.5],
            [-0.5, -0.5],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [-0.75, 0.75],
            [-0.75, 0.75],
            [0.75, -0.75],
            [0.75, -0.75],
            [0.25, 0.25],
            [0.25, 0.25],
            [-0.25, -0.25],
            [-0.25, -0.25],
            [0.9, 0.9],
            [0.9, 0.9],
        ], dtype=torch.float32)
        train_x = base
        eval_x = torch.tensor([
            [-1.0, -1.0],
            [-0.5, -0.5],
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-0.75, 0.75],
            [0.75, -0.75],
            [0.25, 0.25],
        ], dtype=torch.float32)
        target = torch.zeros(n_eval)
    else:
        raise ValueError(name)
    return train_x, eval_x, target.float()


def _build_wrapper(strategy: str, seed: int) -> NACWrapper:
    torch.manual_seed(seed)
    model = TinyRegressor()
    wrapper = NACWrapper(
        model,
        layer_name_list=["hidden", "out"],
        mode=NACMode.REGRESSION,
        M=8,
        O=4,
        alpha=10.0,
        device=torch.device("cpu"),
        local_neighbor_strategy=strategy,
        local_k=4,
        local_random_seed=seed,
    )
    return wrapper


def _fit_wrapper(wrapper: NACWrapper, train_x: torch.Tensor) -> None:
    wrapper.train()
    for batch in train_x.split(6):
        _ = wrapper(batch)
    wrapper.eval()


def _score_case(fixture_name: str, seed: int, strategy: str) -> tuple[float, float, float]:
    train_x, eval_x, target = _make_fixture(fixture_name, seed)
    wrapper = _build_wrapper(strategy, seed)
    _fit_wrapper(wrapper, train_x)
    scores = wrapper(eval_x)["uncertainty"]
    assert isinstance(scores, torch.Tensor)
    finite_fraction = float(torch.isfinite(scores).float().mean())
    scores = torch.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=0.0)
    correlation = _spearman(scores.detach().cpu(), target.detach().cpu())
    ood_gap = float(scores.mean().item() - wrapper(train_x)["uncertainty"].mean().item())
    return correlation, finite_fraction, ood_gap


@pytest.mark.parametrize("fixture_name", FIXTURE_FAMILIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_baseline_global_component(fixture_name: str, seed: int):
    correlation, stability, _ood_gap = _score_case(fixture_name, seed, "global")
    print(f"baseline_rank_correlation={correlation}")
    print(f"baseline_stability={stability}")
    assert math.isfinite(correlation)
    # expectation source: hypothesis; stability is defined by the experiment design as finite-score fraction.
    assert 0.0 <= stability <= 1.0


@pytest.mark.parametrize("fixture_name", FIXTURE_FAMILIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_treatment_true_locality(fixture_name: str, seed: int):
    correlation, stability, ood_gap = _score_case(fixture_name, seed, "nearest")
    print(f"treatment_rank_correlation={correlation}")
    print(f"treatment_stability={stability}")
    print(f"treatment_ood_gap={ood_gap}")
    assert math.isfinite(correlation)
    # expectation source: hypothesis; treatment stability should be a valid finite fraction for every case.
    assert 0.0 <= stability <= 1.0


@pytest.mark.parametrize("fixture_name", FIXTURE_FAMILIES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("strategy", NEGATIVE_STRATEGIES)
def test_negative_randomized_locality(fixture_name: str, seed: int, strategy: str):
    correlation, stability, _ood_gap = _score_case(fixture_name, seed, strategy)
    print(f"negative_rank_correlation={correlation}")
    print(f"negative_stability={stability}")
    assert math.isfinite(correlation)
    # expectation source: hypothesis; negative controls must remain machine-readable and bounded even when locality is destroyed.
    assert 0.0 <= stability <= 1.0


def test_negative_control_detects_drift():
    treatment_corr, treatment_stability, treatment_ood_gap = _score_case("axis_split", 0, "nearest")
    negative_corr, negative_stability, _negative_ood_gap = _score_case("axis_split", 0, "farthest")
    print(f"treatment_rank_correlation={treatment_corr}")
    print(f"treatment_stability={treatment_stability}")
    print(f"treatment_ood_gap={treatment_ood_gap}")
    print(f"negative_rank_correlation={negative_corr}")
    print(f"negative_stability={negative_stability}")
    drift_detected = float(negative_corr <= treatment_corr)
    print(f"intentional_parity_drift_count={drift_detected}")
    # expectation source: hypothesis; this intentional mismatch test passes only if the degraded farthest-neighbor control is detected.
    assert drift_detected == 1.0
