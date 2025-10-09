from typing import Sequence
import math

from .module import Parameter
from ..scalar.scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters
        
    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0, momentum: float = 0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue

            is_scalar = hasattr(p.value, "derivative")
            
            grad = p.value.derivative if is_scalar and p.value.derivative is not None else p.value.grad
            
            if grad is None:
                continue

            if self.momentum == 0.0:
                # Standard SGD
                update_val = self.lr * grad
            else:
                # SGD with momentum
                if p not in self.velocities:
                    self.velocities[p] = 0.0 if is_scalar else grad * 0.0
                
                v = self.velocities[p]
                v_new = self.momentum * v + grad
                self.velocities[p] = v_new
                update_val = self.lr * v_new

            if is_scalar:
                p.update(Scalar(p.value.data - update_val))
            else:
                p.update(p.value - update_val)


class RMSProp(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1e-2, decay_rate: float = 0.9, eps: float = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        self.s_vals = {}

    def step(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue

            is_scalar = hasattr(p.value, "derivative")
            
            grad = p.value.derivative if is_scalar and p.value.derivative is not None else p.value.grad
            
            if grad is None:
                continue

            if p not in self.s_vals:
                if is_scalar:
                    self.s_vals[p] = 0.0
                else:
                    self.s_vals[p] = grad * 0.0
            
            s = self.s_vals[p]

            s_new = self.decay_rate * s + (1 - self.decay_rate) * (grad * grad)
            self.s_vals[p] = s_new

            if is_scalar:
                update_val = self.lr * grad / (math.sqrt(s_new) + self.eps)
            else:
                update_val = self.lr * grad / (s_new.sqrt() + self.eps)

            if is_scalar:
                p.update(Scalar(p.value.data - update_val))
            else:
                p.update(p.value - update_val)
