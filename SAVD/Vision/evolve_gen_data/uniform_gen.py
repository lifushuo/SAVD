def get_random_problems(batch_size, problem_size):
    import torch
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems