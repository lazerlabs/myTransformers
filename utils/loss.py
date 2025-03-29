import torch
import torch.nn as nn
import torch.nn.functional as F

class StockPredictionLoss:
    """
    Collection of loss functions for stock market prediction
    """
    @staticmethod
    def mse_loss(y_pred, y_true, mask=None):
        """
        Standard Mean Squared Error
        Good for general prediction tasks
        """
        if mask is not None:
            return torch.mean(mask * (y_pred - y_true) ** 2)
        return F.mse_loss(y_pred, y_true)

    @staticmethod
    def mae_loss(y_pred, y_true, mask=None):
        """
        Squared Mean Absolute Error: (MAE)^2
        Gives more importance to small errors compared to MSE
        """
        mae = torch.abs(y_pred - y_true).mean()
        return mae

    @staticmethod
    def squared_mae_loss(y_pred, y_true, mask=None):
        """
        Squared Mean Absolute Error: (MAE)^2
        Gives more importance to small errors compared to MSE
        """
        mae = torch.abs(y_pred - y_true).mean()
        return mae ** 2

    @staticmethod
    def huber_loss(y_pred, y_true, delta=1.0, mask=None):
        """
        Huber Loss: Combines MSE and MAE
        Less sensitive to outliers than MSE, but more than MAE
        Good for financial data with occasional large moves
        """
        return F.huber_loss(y_pred, y_true, delta=delta)

    @staticmethod
    def asymmetric_loss(y_pred, y_true, alpha=1.5, mask=None):
        """
        Asymmetric Loss: Penalizes under-prediction more than over-prediction
        Useful when missing upside moves is more costly than missing downside moves
        alpha > 1: penalizes under-prediction more
        alpha < 1: penalizes over-prediction more
        """
        diff = y_pred - y_true
        loss = torch.where(diff > 0, diff**2, (alpha * diff)**2)
        return torch.mean(loss)

    @staticmethod
    def directional_loss(y_pred, y_true, base_loss="mse", direction_weight=0.2, mask=None):
        """
        Combined loss that considers both value accuracy and direction accuracy
        Helps ensure the model predicts price movements correctly
        """
        # Base value loss (MSE by default)
        if base_loss == "mse":
            value_loss = F.mse_loss(y_pred, y_true)
        else:
            value_loss = F.l1_loss(y_pred, y_true) ** 2

        # Direction loss
        pred_direction = torch.sign(y_pred[..., 1:] - y_pred[..., :-1])
        true_direction = torch.sign(y_true[..., 1:] - y_true[..., :-1])
        direction_loss = torch.mean((pred_direction - true_direction) ** 2)

        return value_loss + direction_weight * direction_loss

    @staticmethod
    def adaptive_scale_loss(y_pred, y_true, alpha=0.3, beta=2.0, mask=None):
        """
        Adaptive Scale Loss: Gives more importance to small price movements while handling larger ones
        
        Args:
            y_pred: predicted values
            y_true: true values
            alpha: weight for the small movement component (default: 0.3)
            beta: exponential scaling factor for larger movements (default: 2.0)
            mask: optional mask for masked loss computation
        
        The loss combines three components:
        1. Squared error scaled by 1/abs(y_true) to make small changes more important
        2. Direction accuracy with higher weight for small movements
        3. Regular MSE for stability
        """
        # Compute relative error (scaled by true value magnitude)
        relative_diff = (y_pred - y_true) / (torch.abs(y_true) + 1e-5)
        relative_loss = torch.mean(relative_diff ** 2)
        
        # Direction component with adaptive weighting
        pred_diff = y_pred[..., 1:] - y_pred[..., :-1]
        true_diff = y_true[..., 1:] - y_true[..., :-1]
        
        # Normalize differences to [-1, 1] range
        pred_diff_norm = torch.tanh(pred_diff)
        true_diff_norm = torch.tanh(true_diff)
        
        # Weight smaller movements more heavily
        movement_scale = 1.0 / (torch.abs(true_diff) + 1e-5)
        direction_loss = torch.mean(movement_scale * (pred_diff_norm - true_diff_norm) ** 2)
        
        # Regular MSE for stability
        mse_loss = F.mse_loss(y_pred, y_true)
        
        # Combine losses with exponential scaling
        combined_loss = (
            alpha * relative_loss +
            (1 - alpha) * direction_loss +
            torch.pow(mse_loss, beta)
        )
        
        return combined_loss

def get_loss_function(loss_type="mse", **kwargs):
    """
    Factory function to get the desired loss function
    
    Args:
        loss_type: string indicating which loss function to use
        **kwargs: additional arguments for specific loss functions
    
    Returns:
        loss_function: callable loss function
    """
    loss_functions = {
        "mse": StockPredictionLoss.mse_loss,
        "squared_mae": StockPredictionLoss.squared_mae_loss,
        "huber": lambda y_pred, y_true, mask=None: StockPredictionLoss.huber_loss(
            y_pred, y_true, delta=kwargs.get('delta', 1.0), mask=mask
        ),
        "asymmetric": lambda y_pred, y_true, mask=None: StockPredictionLoss.asymmetric_loss(
            y_pred, y_true, alpha=kwargs.get('alpha', 1.5), mask=mask
        ),
        "directional": lambda y_pred, y_true, mask=None: StockPredictionLoss.directional_loss(
            y_pred, y_true, 
            base_loss=kwargs.get('base_loss', 'mse'),
            direction_weight=kwargs.get('direction_weight', 0.2),
            mask=mask
        ),
        "adaptive": lambda y_pred, y_true, mask=None: StockPredictionLoss.adaptive_scale_loss(
            y_pred, y_true,
            alpha=kwargs.get('alpha', 0.3),
            beta=kwargs.get('beta', 2.0),
            mask=mask
        )
    }

    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Available types: {list(loss_functions.keys())}")

    return loss_functions[loss_type]

# Example usage:
"""
# In training script:
loss_fn = get_loss_function(
    loss_type="directional",
    base_loss="squared_mae",
    direction_weight=0.3
)

# Then use it:
loss = loss_fn(predictions, targets)
""" 