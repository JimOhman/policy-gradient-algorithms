from agent import VPGAgent
import numpy as np
import gym
from args import parse_args
from logger import Logger
from networks import SimpleNet
from torch.optim import RMSprop, Adam
import datetime
import pytz
import time


def get_env(args):
  env = gym.make(args['environment'])
  return env

def get_net(env, args):
  state_space = env.observation_space.shape[0]
  action_space = env.action_space.n
  if args['architecture'] == 'SimpleNet':
    net = SimpleNet(action_space=action_space, state_space=state_space)
  else:
    raise NotImplementedError
  return net

def get_opt(net, args):
  if args['optimizer'] == 'Adam':
    optimizer = Adam(net.parameters(), lr=args['learning_rate'], eps=0.00015)
  elif args['optimizer'] == 'RMSprop':
    optimizer = RMSprop(net.parameters(), lr=args['learning_rate'], momentum=0.95, eps=0.01)
  else:
    raise NotImplementedError
  return optimizer

def get_agent(net, optimizer, env, args):
    agent = VPGAgent(net=net, optimizer=optimizer, env=env, args=args)
  return agent

def train(agent, args, logr=None):
  verbose = args['verbose']
  total_batches = args['total_batches']
  render = args['render_env']
  print("\nStarting training towards {} batch updates. \n".format(total_batches))
  for batch_num in range(total_batches):
    batch_loss, batch_rets, batch_lens = agent.train_one_batch(render=render)

    if verbose:
      date_now = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")
      batch_stats = (round(np.mean(batch_lens)), round(np.std(batch_lens)),
                     round(np.mean(batch_rets)), round(np.std(batch_rets)))

      print("[{}] ({}/{}) --> length: {} ({}), return: {} ({})".format(date_now,
                                                                       batch_num,
                                                                       total_batches,
                                                                       *batch_stats))
    if logr is not None:
      logr.add_value(tag='loss', value=batch_loss.detach(), it=batch_num)
      logr.add_value(tag='return', value=np.mean(batch_rets), it=batch_num)
      logr.add_value(tag='length', value=np.mean(batch_lens), it=batch_num)

    batch_num += 1
  print("\nFinished training.")

def init(args):
  print()
  env = get_env(args=args)
  print("Using environment: {}".format(args['environment']))
  net = get_net(env=env, args=args)
  print("1. Created net: {}".format(args['architecture']))
  optimizer = get_opt(net=net, args=args)
  print("2. Created optimizer: {} with lr = {}".format(args['optimizer'], args['learning_rate']))
  agent = get_agent(net=net, optimizer=optimizer, env=env, args=args)
  print("3. Assembled agentrithm: {}".format(args['agentrithm']))
  time.sleep(1.)
  return env, net, optimizer, agent

def evaluate(agent, args):
  render = args['render_env']
  eps_rews = [], eps_obs = []
  for episode_num in range(args['eval_episodes']):
    ep_rews, ep_obs, _ = agent.run_one_episode(render=render)
    print("Batch {} ==> return: {}, length = {}".format(sum(ep_rews), len(ep_rews)))
    eps_rews.append(ep_rews)
    eps_obs.append(ep_obs)

if __name__ == '__main__':
  args = parse_args()
  run_tag = '|' + 'opt-' + args['optimizer'] + '|' + 'lr-' + str(args['learning_rate'])
  logr = Logger(args=args, run_tag=run_tag)
  env, net, optimizer, agent = init(args)

  if args['mode'] == 'train':
    train(agent=agent, args=args, logr=logr)
  elif args['mode'] == 'evaluate':
    evaluate(net=net, args=args)
  else:
    raise NotImplementedError

