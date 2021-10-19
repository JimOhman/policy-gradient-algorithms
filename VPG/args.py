import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture', type=str, default='SimpleNet')
    parser.add_argument('--algorithm', type=str, default='VPG')

    parser.add_argument('--mode', choices=['train', 'evaluate'], type=str, default='train')
    parser.add_argument('--environment', type=str, default='CartPole-v1')

    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--gamma', type=int, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_batches', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='RMSprop')

    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--render_state', action='store_true')
    parser.add_argument('--render_env', action='store_true')
    parser.add_argument('--eval_episodes', type=int, default=1)

    args = parser.parse_args()
    return vars(args)
