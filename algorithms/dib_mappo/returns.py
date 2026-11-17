"""Same native n-step terms and addition order, batched over start times."""
import torch


def nstep_returns(args, rewards, mask, values, nsteps):
    length = rewards.shape[1]
    result = torch.zeros_like(values[:, :-1])
    for step in range(min(nsteps, length - 1) + 1):
        count = length - step
        source = values if step == nsteps else rewards
        result[:, :count] += args.gamma ** step * source[:, step:length] * mask[:, step:length]
        if step < nsteps and args.add_value_last_step:
            result[:, count - 1] += args.gamma ** (step + 1) * values[:, length]
    return result
