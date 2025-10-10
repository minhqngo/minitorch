from .nn import logsoftmax


def mse_loss(y_pred, y_true):
    """
    Mean Squared Error Loss.

    Args:
        y_pred (Tensor): Predicted values, shape (batch_size, 1).
        y_true (Tensor): True values, shape (batch_size, 1).

    Returns:
        Tensor: The mean squared error loss.
    """
    diff = y_pred - y_true
    return (diff * diff).sum() / y_pred.shape[0]


def nll_loss(y_pred_log_probs, y_true):
    """
    Negative Log-Likelihood Loss.

    Args:
        y_pred_log_probs (Tensor): Log-probabilities of predictions, shape (batch_size, num_classes).
        y_true (Tensor): True class indices, shape (batch_size,).

    Returns:
        Tensor: The negative log-likelihood loss.
    """
    batch_size, num_classes = y_pred_log_probs.shape
    
    # Create one-hot encoded tensor for y_true
    y_one_hot = y_pred_log_probs.zeros(y_pred_log_probs.shape)
    y_one_hot.requires_grad_(False)

    for i in range(batch_size):
        y_one_hot[i, int(y_true[i].item())] = 1

    loss = -(y_pred_log_probs * y_one_hot).sum()
    return loss / batch_size


def cross_entropy_loss(y_pred_logits, y_true):
    """
    Cross-Entropy Loss.

    Args:
        y_pred_logits (Tensor): Raw logits from the model, shape (batch_size, num_classes).
        y_true (Tensor): True class indices, shape (batch_size,).

    Returns:
        Tensor: The cross-entropy loss.
    """
    log_probs = logsoftmax(y_pred_logits, dim=1)
    return nll_loss(log_probs, y_true)


def bce_loss(y_pred, y_true):
    """
    Binary Cross-Entropy Loss.

    Args:
        y_pred (Tensor): Predicted probabilities, shape (batch_size, 1).
        y_true (Tensor): True labels (0 or 1), shape (batch_size, 1).

    Returns:
        Tensor: The binary cross-entropy loss.
    """
    loss = -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())
    return loss.sum() / y_pred.shape[0]
