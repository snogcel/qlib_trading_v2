
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Callable, Dict, List, Sequence, cast

from collections import defaultdict

from tianshou.policy import BasePolicy

from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.utils import FiniteEnvType, LogWriter, LogBuffer

from qlib.rl.trainer import Trainer
from qlib.rl.trainer.callbacks import Callback
from qlib_custom.custom_training_vessel import CustomTrainingVessel

class CustomTrainer(Trainer):    
    def __init__(
        self,
        **trainer_kwargs
    ):
        self.current_episode = 0
        if 'logger' in trainer_kwargs:
            self.logger = trainer_kwargs.pop('logger')
        super().__init__(**trainer_kwargs)

    def _stage_prefix(self) -> str:
        return "val" if self.current_stage == "val" else "train"

    def _metrics_callback(self, on_episode: bool, on_collect: bool, log_buffer: LogBuffer) -> None:               
        if on_episode:
            self.current_episode = log_buffer.global_episode
            self.logger.set_step(self.current_episode)
            raw_metrics = log_buffer.episode_metrics()
        elif on_collect:
            raw_metrics = log_buffer.collect_metrics()
        else:
            return  # No metrics to process

        metrics = {f"{self._stage_prefix()}/{name}": value for name, value in raw_metrics.items()}
        self.logger.log_scalars(metrics)
        self.metrics.update(metrics)


def train(
    simulator_fn: Callable[[InitialStateType], Simulator],
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    initial_states: Sequence[InitialStateType],
    policy: BasePolicy,
    reward: Reward,
    vessel_kwargs: Dict[str, Any],
    trainer_kwargs: Dict[str, Any],
) -> None:
    """Train a policy with the parallelism provided by RL framework.

    Experimental API. Parameters might change shortly.

    Parameters
    ----------
    simulator_fn
        Callable receiving initial seed, returning a simulator.
    state_interpreter
        Interprets the state of simulators.
    action_interpreter
        Interprets the policy actions.
    initial_states
        Initial states to iterate over. Every state will be run exactly once.
    policy
        Policy to train against.
    reward
        Reward function.
    vessel_kwargs
        Keyword arguments passed to :class:`TrainingVessel`, like ``episode_per_iter``.
    trainer_kwargs
        Keyword arguments passed to :class:`Trainer`, like ``finite_env_type``, ``concurrency``.
    """

    vessel = CustomTrainingVessel(
        simulator_fn=simulator_fn,
        state_interpreter=state_interpreter,
        action_interpreter=action_interpreter,
        policy=policy,
        train_initial_states=initial_states,
        reward=reward,  # ignore none
        **vessel_kwargs,
    )
    trainer = CustomTrainer(**trainer_kwargs)
    trainer.fit(vessel)

def backtest(
    simulator_fn: Callable[[InitialStateType], Simulator],
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    initial_states: Sequence[InitialStateType],
    policy: BasePolicy,
    logger: LogWriter | List[LogWriter],
    reward: Reward | None = None,
    **trainer_kwargs,
) -> None:
    """Backtest with the parallelism provided by RL framework.

    Experimental API. Parameters might change shortly.

    Parameters
    ----------
    simulator_fn
        Callable receiving initial seed, returning a simulator.
    state_interpreter
        Interprets the state of simulators.
    action_interpreter
        Interprets the policy actions.
    initial_states
        Initial states to iterate over. Every state will be run exactly once.
    policy
        Policy to test against.
    logger
        Logger to record the backtest results. Logger must be present because
        without logger, all information will be lost.
    reward
        Optional reward function. For backtest, this is for testing the rewards
        and logging them only.
    finite_env_type
        Type of finite env implementation.
    concurrency
        Parallel workers.
    """

    vessel = CustomTrainingVessel(
        simulator_fn=simulator_fn,
        state_interpreter=state_interpreter,
        action_interpreter=action_interpreter,
        policy=policy,
        test_initial_states=initial_states,
        reward=cast(Reward, reward),  # ignore none
    )
    trainer = CustomTrainer(trainer_kwargs)
    trainer.test(vessel)
