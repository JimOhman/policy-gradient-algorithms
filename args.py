import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Model specific
    parser.add_argument('--architecture', type=str, default='SimpleNet')
    parser.add_argument('--algorithm', type=str, default='VPG')
    # parser.add_argument('--load_checkpoint', type=str, default='')
    # parser.add_argument('--load_model', type=str, default='')

    # Modes are train_dqn or evaluate_dqn
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--environment', type=str, default='CartPole-v1')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--gamma', type=int, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_batches', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='RMSprop')

    # Compute specific
    #parser.add_argument('--multi_gpu', action='store_true')
    #parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--verbose', action='store_true')

    # Evaluation specific
    parser.add_argument('--render_state', action='store_true')
    parser.add_argument('--render_env', action='store_true')
    parser.add_argument('--eval_episodes', type=int, default=1)

    args = parser.parse_args()
    return vars(args)