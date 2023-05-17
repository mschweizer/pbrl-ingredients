import functools

# noinspection PyUnresolvedReferences
import beach_walk_env
import gym
from imitation.scripts.common import common, reward, rl
from imitation.scripts.common.common import common_ingredient
from imitation.scripts.common.reward import reward_ingredient
from imitation.scripts.common.rl import rl_ingredient
from imitation.scripts.train_preference_comparisons import save_checkpoint
from sacred import Experiment
from stable_baselines3 import PPO

from pbrl_ingredients.agent import pbrl_agent_ingredient, create_agent
from pbrl_ingredients.fragmenter import fragmenter_ingredient, create_fragmenter
from pbrl_ingredients.gatherer import gatherer_ingredient, create_gatherer
from pbrl_ingredients.preference_model import preference_model_ingredient, create_preference_model
from pbrl_ingredients.reward_trainer import reward_trainer_ingredient, create_reward_trainer
from pbrl_ingredients.trajectory_generator import trajectory_generator_ingredient, create_trajectory_generator

pbrl_experiment = Experiment("pbrl_experiment",
                             ingredients=[common_ingredient,
                                          fragmenter_ingredient,
                                          gatherer_ingredient,
                                          preference_model_ingredient,
                                          reward_trainer_ingredient,
                                          trajectory_generator_ingredient,
                                          pbrl_agent_ingredient,
                                          reward_ingredient,
                                          rl_ingredient])


def apply_obs_wrappers(env: gym.Env, env_idx: int) -> gym.Env:
    env = FullyObsWrapper(env)
    env = CustomObsWrapper(env)
    return env


@common_ingredient.named_config
def wrapped_environment():
    post_wrappers = [apply_obs_wrappers]


# noinspection PyUnusedLocal
@pbrl_experiment.config
def cfg():
    total_timesteps = int(1e5)
    total_comparisons = 500
    checkpoint_interval = 0  # Num epochs between saving (<0 disables, =0 final only)
    trajectory_path = None
    save_preferences = False


@pbrl_experiment.automain
def run(total_timesteps: int,
        total_comparisons: int,
        checkpoint_interval: int,
        ):
    """
    Args:
        total_timesteps:
        total_comparisons:
        checkpoint_interval: Save the reward model and policy models (if
            trajectory_generator contains a policy) every `checkpoint_interval`
            iterations and after training is complete. If 0, then only save weights
            after training is complete. If <0, then don't save weights at all.

    Returns:
         None
    """
    custom_logger, log_dir = common.setup_logging()
    rng = common.make_rng()

    with common.make_venv(post_wrappers=[apply_obs_wrappers]) as venv:

        reward_net = reward.make_reward_net(venv)
        relabel_reward_fn = functools.partial(
            reward_net.predict_processed,
            update_stats=False,
        )
        rl_agent = rl.make_rl_algo(venv=venv, rl_cls=PPO, relabel_reward_fn=relabel_reward_fn)
        trajectory_generator = create_trajectory_generator(rl_agent, reward_net, venv, rng, custom_logger)
        fragmenter = create_fragmenter(custom_logger, rng)
        gatherer = create_gatherer(rng)
        preference_model = create_preference_model(reward_net)
        reward_trainer = create_reward_trainer(preference_model, rng)

        pbrl_agent = create_agent(trajectory_generator=trajectory_generator,
                                  fragmenter=fragmenter,
                                  gatherer=gatherer,
                                  reward_trainer=reward_trainer,
                                  reward_net=reward_net,
                                  custom_logger=custom_logger)

        def save_callback(iteration_num):
            if checkpoint_interval > 0 and iteration_num % checkpoint_interval == 0:
                save_checkpoint(
                    trainer=pbrl_agent,
                    save_path=log_dir / "checkpoints" / f"{iteration_num:04d}",
                    allow_save_policy=True,
                )

        pbrl_agent.train(
            total_timesteps,
            total_comparisons,
            callback=save_callback,
        )

        # Save final artifacts
        if checkpoint_interval >= 0:
            save_checkpoint(
                trainer=pbrl_agent,
                save_path=log_dir / "checkpoints" / "final",
                allow_save_policy=True,
            )
