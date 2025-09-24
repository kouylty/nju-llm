import sys
sys.path.insert(0, ".")

import pytest

import torch
from torch.testing import assert_close
from src import ManualLinear
from src import AutogradLinear

# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float32


# configs for each toy test case
toy_test_cases = {
    "case1": {
        "b": 32,
        "h": 33,
        "d": 34,
        "e": 35,
    }
}


@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases.items(),
)
def test_task1(case_key, case_config):
    # define hyper parameters
    b, h, d, e = case_config["b"], case_config["h"], case_config["d"], case_config["e"]
    seed = case_config.pop("seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    device, dtype = case_config.pop("device", DEVICE), case_config.pop("dtype", DTYPE)

    # construct the necessary tensors
    torch.manual_seed(seed)

    # ========== Manual ==========
    x = torch.randn(b, h, d, device=device, dtype=dtype)
    manual_linear = ManualLinear(d, e, device=device, dtype=dtype)
    
    #----- test if the function works without grad_output -----#
    output = manual_linear.forward(x)
    grad_output = 2 * output
    grad_input = manual_linear.backward(grad_output)
    grad_weight = manual_linear.W_grad

    # ========== Autograd ==========
    autograd_linear = AutogradLinear(d, e, device=device, dtype=dtype)
    with torch.no_grad():
        autograd_linear.W.copy_(manual_linear.W)

    x_autograd = x.clone().detach().requires_grad_()
    output_ref = autograd_linear(x_autograd)
    loss_autograd = output_ref.pow(2).sum()
    loss_autograd.backward()
    grad_input_ref = x_autograd.grad
    grad_weight_ref = autograd_linear.W.grad
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    # check if the grad_input tensor is correct
    assert grad_input is not None, "grad_input should not be None"
    # check if the grad_weight tensor is correct 
    assert grad_weight is not None, "grad_weight should not be None"
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    # check if the grad_input tensor is correct
    assert_close(grad_input, grad_input_ref, atol=atol, rtol=rtol)
    # check if the grad_weight tensor is correct
    assert_close(grad_weight, grad_weight_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main()
    
