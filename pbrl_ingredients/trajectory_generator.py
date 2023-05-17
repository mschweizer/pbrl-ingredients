from imitation.algorithms import preference_comparisons
from sacred import Ingredient

trajectory_generator_ingredient = Ingredient("trajectory_generator")


@trajectory_generator_ingredient.config
def cfg():
    exploration_fraction = 0.1


@trajectory_generator_ingredient.capture
def create_trajectory_generator(rl_agent, reward_net, env, rng, custom_logger, exploration_fraction, kwargs=None):
    preference_comparisons.AgentTrainer(
        algorithm=rl_agent,
        reward_fn=reward_net,
        venv=env,
        exploration_frac=exploration_fraction,
        rng=rng,
        custom_logger=custom_logger,
        **kwargs if kwargs else {},
    )