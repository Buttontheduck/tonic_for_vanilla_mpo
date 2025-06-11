'''Script used to train agents.'''

import argparse
import os

import tonic
import tonic.torch
import yaml
import wandb

from omegaconf import OmegaConf
from tonic.utils.trainer import Trainer as trainer_fun
from tonic.torch.agents.mpo import MPO 



def train(wandb_config,trainer_config,agent_config,
    header, agent, environment, test_environment, trainer, before_training,
    after_training, parallel, sequential, seed, name, environment_name,
    checkpoint, path
):
    '''Trains an agent on an environment.'''

    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    for key in ('wandb_config', 'trainer_config', 'agent_config'):
        args.pop(key, None)
    
    wandb_params = wandb_config['wandb_config']
    trainer_params = trainer_config['trainer_config']
    agent_params = agent_config['agent_config']


    try:
        checkpoint_path = None

        # Process the checkpoint path same way as in tonic.play
        if path:
            tonic.logger.log(f'Loading experiment from {path}')

            # Use no checkpoint, the agent is freshly created.
            if checkpoint == 'none' or agent is not None:
                tonic.logger.log('Not loading any weights')

            else:
                checkpoint_path = os.path.join(path, 'checkpoints')
                if not os.path.isdir(checkpoint_path):
                    tonic.logger.error(f'{checkpoint_path} is not a directory')
                    checkpoint_path = None

                # List all the checkpoints.
                checkpoint_ids = []
                for file in os.listdir(checkpoint_path):
                    if file[:5] == 'step_':
                        checkpoint_id = file.split('.')[0]
                        checkpoint_ids.append(int(checkpoint_id[5:]))

                if checkpoint_ids:
                    # Use the last checkpoint.
                    if checkpoint == 'last':
                        checkpoint_id = max(checkpoint_ids)
                        checkpoint_path = os.path.join(
                            checkpoint_path, f'step_{checkpoint_id}')

                    # Use the specified checkpoint.
                    else:
                        checkpoint_id = int(checkpoint)
                        if checkpoint_id in checkpoint_ids:
                            checkpoint_path = os.path.join(
                                checkpoint_path, f'step_{checkpoint_id}')
                        else:
                            tonic.logger.error(f'Checkpoint {checkpoint_id} '
                                               f'not found in {checkpoint_path}')
                            checkpoint_path = None

                else:
                    tonic.logger.error(f'No checkpoint found in {checkpoint_path}')
                    checkpoint_path = None

            # Load the experiment configuration.
            arguments_path = os.path.join(path, 'config.yaml')
            with open(arguments_path, 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
            config = argparse.Namespace(**config)

            header = header or config.header
            agent = agent or config.agent
            environment = environment or config.test_environment
            environment = environment or config.environment
            trainer = trainer or config.trainer

        # Run the header first, e.g. to load an ML framework.
        if header:
            exec(header)

        # Build the training environment.
        _environment = environment
        environment = tonic.environments.distribute(
            lambda: eval(_environment), parallel, sequential)
        environment.initialize(seed=seed)

        # Build the testing environment.
        _test_environment = test_environment if test_environment else _environment
        test_environment = tonic.environments.distribute(
            lambda: eval(_test_environment))
        test_environment.initialize(seed=seed + 10000)

        # Build the agent.
        if not agent:
            raise ValueError('No agent specified.')
        agent = MPO(**agent_params)
        #agent = eval(agent) Removed because of hydra conf
        
        agent.initialize(
            observation_space=environment.observation_space,
            action_space=environment.action_space, seed=seed)

        # Load the weights of the agent form a checkpoint.
        if checkpoint_path:
            agent.load(checkpoint_path)

        # Initialize the logger to save data to the path environment/name/seed.
        if not environment_name:
            if hasattr(test_environment, 'name'):
                environment_name = test_environment.name
            else:
                environment_name = test_environment.__class__.__name__
        if not name:
            if hasattr(agent, 'name'):
                name = agent.name
            else:
                name = agent.__class__.__name__
            if parallel != 1 or sequential != 1:
                name += f'-{parallel}x{sequential}'
        path = os.path.join(environment_name, name, str(seed))
        tonic.logger.initialize(path, script_path=__file__, config=args)

        # TODO: make init_config to added wandb as wandb.init(**wandb_params, config=init_config)
        #init_config = OmegaConf.to_container(agent_params, resolve=True)
        wandb.init(**wandb_params)
        
        # Build the trainer.
        trainer = trainer or 'tonic.Trainer()'
        trainer = eval(trainer)
        trainer = trainer_fun(**trainer_params)
        trainer.initialize(
            agent=agent, environment=environment,
            test_environment=test_environment)

        # Run some code before training.
        if before_training:
            exec(before_training)

        # Train.
        trainer.run()

        # Run some code after training.
        if after_training:
            exec(after_training)
    finally:
        wandb.finish()


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--header')
    parser.add_argument('--agent')
    parser.add_argument('--environment', '--env')
    parser.add_argument('--test_environment', '--test_env')
    parser.add_argument('--trainer')
    parser.add_argument('--before_training')
    parser.add_argument('--after_training')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--sequential', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name')
    parser.add_argument('--environment_name')
    parser.add_argument('--checkpoint', default='last')
    parser.add_argument('--path')

    args = vars(parser.parse_args())
    train(**args)
