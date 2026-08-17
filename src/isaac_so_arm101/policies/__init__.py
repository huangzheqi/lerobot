"""Policy helpers for SO-ARM101 tasks."""

__all__ = ["ActActorCritic"]


def __getattr__(name):
    if name == "ActActorCritic":
        from .act_actor_critic import ActActorCritic

        return ActActorCritic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
