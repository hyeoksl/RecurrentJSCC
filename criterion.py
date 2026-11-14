import torch.nn.functional as F
from torch import Tensor, nn


class CustomJSCCLoss(nn.Module):
    def __init__(self, loss_type: str = "MSE") -> None:
        super().__init__()
        if loss_type.upper() != "MSE":
            raise ValueError(f"Unsupported loss type: {loss_type}; expected MSE")

    def forward(self, output: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        mse = F.mse_loss(output["x_hat"].float(), target)
        return {"loss": mse, "MSE": mse}
