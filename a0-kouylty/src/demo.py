import torch

# change here, by kouylty, at 20250907
# ========== ManualLinear ==========
class ManualLinear:
    def __init__(self, in_dim, out_dim, device=None, dtype=torch.float32):
        self.W = torch.randn(in_dim, out_dim, device=device, dtype=dtype, requires_grad=False)
        self.W_grad = torch.zeros_like(self.W)

    def forward(self, x):
        self.input = x
        # [b,h,d] @ [d,e] -> [b,h,e]
        return x @ self.W

    def backward(self, grad_output):
        # grad_output: [b, h, e]
        # dL/dW = x^T @ grad_output
        b, h, d = self.input.shape
        # [b,h,d] -> [b*h, d]
        x_flat = self.input.reshape(-1, d)
        # [b,h,e] -> [b*h, e]
        grad_out_flat = grad_output.reshape(-1, self.W.shape[1])
        # [d, b*h] @ [b*h, e] -> [d, e]
        self.W_grad = x_flat.T @ grad_out_flat
        # dL/dx = grad_output @ W^T
        # [b,h,e] @ [e,d] -> [b,h,d]
        grad_input = grad_output @ self.W.T
        return grad_input

    def step(self, lr=1e-2):
        self.W -= lr * self.W_grad


# ========== AutogradLinear ==========
class AutogradLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device=None, dtype=torch.float32):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(in_dim, out_dim, device=device, dtype=dtype))

    def forward(self, x):
        return x @ self.W


# ========== test ==========
def test_manual_vs_autograd_linear():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # param
    b, h, d, e = 2, 3, 4, 5
    x = torch.randn(b, h, d, device=device, dtype=dtype)

    # ========== Manual ==========
    manual_linear = ManualLinear(d, e, device=device, dtype=dtype)
    y_manual = manual_linear.forward(x)
    loss_manual = y_manual.pow(2).sum()
    grad_output = 2 * y_manual
    grad_input_manual = manual_linear.backward(grad_output)
    # manual_linear.step(lr=1e-2)

    # ========== Autograd ==========
    autograd_linear = AutogradLinear(d, e, device=device, dtype=dtype)
    with torch.no_grad():
        autograd_linear.W.copy_(manual_linear.W)

    x_autograd = x.clone().detach().requires_grad_()
    y_autograd = autograd_linear(x_autograd)
    loss_autograd = y_autograd.pow(2).sum()
    loss_autograd.backward()
    grad_input_autograd = x_autograd.grad
    grad_weight_autograd = autograd_linear.W.grad

    torch.testing.assert_close(y_manual, y_autograd, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_input_manual, grad_input_autograd, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(manual_linear.W_grad, grad_weight_autograd, rtol=1e-4, atol=1e-4)

    print("ManualLinear and AutogradLinear match in forward and backward computations.")


if __name__ == '__main__':
    test_manual_vs_autograd_linear()
