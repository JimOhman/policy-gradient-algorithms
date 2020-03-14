import torch
from torch.distributions import Categorical
import numpy as np


class VPG(object):
	def __init__(self, net, optimizer, env, args):
		self.net = net
		self.env = env
		self.optimizer = optimizer
		self.args = args


	def get_policy(self, obs):
		logits = self.net(obs)
		return Categorical(logits=logits)


	def get_action(self, obs):
		return self.get_policy(obs).sample().item()


	def _compute_loss(self, obs, act, weights, reduction=True):
		logp = self.get_policy(obs).log_prob(act)
		loss = -(logp * weights)
		return loss.mean() if reduction else loss


	def _reward_to_go(self, rews):
		n = len(rews)
		rtgs = np.zeros_like(rews)
		for i in reversed(range(n)):
			rtgs[i] = rews[i] + (rtgs[i+1] if i + 1 < n else 0)
		return rtgs


	def run_one_episode(self, render=False):
		obs = self.env.reset()
		done = False

		ep_acts = []
		ep_rews = []
		ep_obs = []

		while not done:
			if render:
				self.env.render()

			ep_obs.append(obs.copy())

			act = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
			obs, rew, done, _ = self.env.step(act)

			ep_acts.append(act)
			ep_rews.append(rew)

		return ep_rews, ep_obs, ep_acts


	def train_one_batch(self, render=False):
		batch_size = self.args['batch_size']
		batch_obs = []
		batch_acts = []
		batch_weights = []
		batch_rets = []
		batch_lens = []

		obs = self.env.reset()
		done = False
		ep_rews = []

		finished_rendering_this_epoch = False

		while True:

			ep_rews, ep_obs, ep_acts = self.run_one_episode(render=render)

			batch_acts += ep_acts
			batch_obs += ep_obs
			batch_weights += list(self._reward_to_go(ep_rews))

			ep_ret, ep_len = sum(ep_rews), len(ep_rews)
			batch_rets.append(ep_ret)
			batch_lens.append(ep_len)

			if len(batch_obs) > batch_size:
				break

		self.optimizer.zero_grad()
		batch_loss = self._compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
								 	    act=torch.as_tensor(batch_acts, dtype=torch.float32),
								        weights=torch.as_tensor(batch_weights, dtype=torch.float32))
		batch_loss.backward()
		self.optimizer.step()
		return batch_loss, batch_rets, batch_lens


