from gym.envs.registration import registry, register, make, spec

#with vision and noise
register(
    id='PendulumSai-v0',
    entry_point='envs.myenvs:PendulumNewV0Env',
    timestep_limit=200,
)


#with only noise no vision
register(
    id='PendulumSai-v1',
    entry_point='envs.myenvs:PendulumNewV1Env',
    timestep_limit=200,
)


