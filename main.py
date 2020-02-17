from models import VPG
from torch.distributions import Categorical
import numpy as np


logits_net = lambda x: x


def get_policy(obs):
	logits = logits_net(obs)
	return Categorical(logits=logits)


def get_action(obs):
	return get_policy(obs).sample().item()


def compute_loss(obs, act, weights):
	logp = get_policy(obs).log_prob(act)
	return -(logp * weights).mean()


def reward_to_go(rews):
	n = len(rews)
	rtgs = np.zeros_like(rews)
	for i in reversed(range(n)):
		rtgs[i] = rews[i] + (rtgs[i+1] if i + 1 < n else 0)
	return rtgs


def train_one_epoch():
	batch_obs = []
	batch_acts = []
	batch_weights = []
	batch_rets = []
	batch_lens = []

	obs = env.reset()
	done = False
	ep_rews = []

	finished_rendering_this_epoch = False

	while True:

		if (not finished_rendering_this_epoch) and render:
			env.render()

		batch_obs.append(obs.copy())

		act = get_action(torch.as_tensor(obs, dtype=torch.float32))
		obs, rew, done, _ = env.step(act)

		batch_acts.append(act)
		ep_rews.append(rew)

		if done:
			ep_ret, ep_len = sum(ep_rews), len(ep_rews)
			batch_rets.append(ep_ret)
			batch_lens.append(ep_len)

			#batch_weights += [ep_ret] * ep_len
			batch_weights += list(reward_to_go(ep_rews))

			obs, done, ep_rews = env.reset(), False, []

			finished_rendering_this_epoch = True

			if len(batch_obs) > batch_size:
				break

	optimizer.zero_grad()
	batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
							  act=torch.as_tensor(batch_acts, dtype=torch.float32),
							  weights=torch.as_tensor(batch_weights, dtype=torch.float32))
	batch_loss.backward()
	optimizer.step()
	return batch_loss, batch_rets, batch_lens
