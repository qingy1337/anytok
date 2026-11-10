"""Pinned names and enabled profile order for the uniform-pool run."""
PROFILES = ("csproject", "data-pipelines", "definitive", "deployment", "dkeratan3", "gildmember", "github", "group-pytorch", "grufaller", "li004330", "mix345-net", "neural-net-1411", "queues", "remote-worker", "stars", "transformer", "transformers-impl", "trenton")
VOLUME_NAME = "anytok-pool-20k"
REPO_ID = "qingy2024/anytok-pool-smollm2-20k"


def restart_repo_path(step):
    # Handoff restarts must not collide with an immutable weights-only grid save.
    folder = 'checkpoints' if step % 2000 == 0 else 'restarts'
    return f'{folder}/step-{step:06d}'
