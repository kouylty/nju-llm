import sys
sys.path.insert(0, ".")

import pytest

import torch
from torch.testing import assert_close
from src import ManualLinear

# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float32


# configs for each toy test case
toy_test_cases = {
    "case1": {
        "b": 2,
        "h": 3,
        "d": 4,
        "e": 5,
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
    
    # construct the reference output tensor
    if case_key == "case1":
        output_ref = torch.tensor(
            [[[-2.1719, -0.3164,  0.6343,  1.0091, -2.4615],
            [-0.7655,  1.1725,  1.6269, -0.1986, -0.5901],
            [ 1.6450,  1.4338, -0.6187,  2.7282,  1.8161]],

            [[-0.1509, -0.0741,  0.3616,  1.7347, -1.4200],
            [-0.9184, -0.5416,  0.6432, -1.6297, -0.9776],
            [ 0.6727, -0.7460, -1.1312,  1.3461, -0.0415]]]
        ).to(device=device, dtype=dtype)
        grad_input_ref = torch.tensor(
            [[[ -1.7479,   9.7409,  -1.4267,   4.2803],
            [ -3.5400,   3.5653,  -1.1995,  -0.0868],
            [  6.5130,  -5.3862, -12.0604,  -4.6756]],

            [[  1.0253,   4.0553,  -5.0721,   2.1386],
            [ -4.2539,   3.0814,   6.5788,   2.3191],
            [  3.9779,  -1.2700,  -3.0797,   0.5641]]]
        ).to(device=device, dtype=dtype)
        grad_weight_ref = torch.tensor(
            [[  7.1879,  -4.5596, -11.0295,  10.0603,   4.7699],
            [-16.6033,  -3.4980,   8.6425,  -5.7984, -17.9120],
            [ -5.1941,  -6.4984,   0.5149, -16.3271,  -2.7786],
            [ -1.9497,  -5.7636,  -3.2433,  10.9067,  -9.6495]]
        ).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    #----- test if the function works without grad_output -----#
    output = manual_linear.forward(x)
    grad_output = 2 * output
    grad_input = manual_linear.backward(grad_output)
    grad_weight = manual_linear.W_grad
    
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
    