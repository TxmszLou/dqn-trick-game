import math
import random

import torch
import torch.nn as nn

from .policies import policy_legal_move
from .replay import Transition


def select_action(
    game,
    policy_net,
    steps_done,
    device,
    eps_start=1,
    eps_end=0.1,
    eps_decay=5000,
    legal_only=False,
):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)

    if sample > eps_threshold:
        with torch.no_grad():
            if legal_only:
                action = policy_legal_move(policy_net, game.get_network_input())
            else:
                action = policy_net(game.get_network_input().to(device)).max(0).indices.view(1, 1)
    else:
        action = torch.tensor([[game.sample_legal_move()]], device=device, dtype=torch.long)

    return action, steps_done + 1, eps_threshold


def optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=100, gamma=0.5):
    transitions = []
    for mem in memory.values():
        if len(mem) >= batch_size:
            transitions += mem.sample(batch_size)
    if transitions == []:
        return None

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = [s for s in batch.next_state if s is not None]

    non_final_next_states = torch.cat(non_final_next_states)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch.to(torch.float)).gather(1, action_batch)

    next_state_values = torch.zeros(len(transitions), device=device)
    if non_final_next_states != []:
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss


def soft_update_target(policy_net, target_net, tau=0.005):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)
