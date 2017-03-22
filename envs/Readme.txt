Observation space is of shape (67603,).
First three are the physical parameter.
From three to end is the image of size 260*260 you can reshape it.
Image is binarized 0 or 1.

Place the folder in keras_rl/ directory.


In your code use 'import envs' to load the new enviroment.
Invoke the environment by 'env = gym.make('PendulumSai-v0')'

All the calls env.render/step/render will print a screen dont worry, no problem.
Calling env.render() will have torque diagram and the others wont.
