import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym
import torch
def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save("./LunarLander_result.gif", writer="pillow", fps = 30)
        
def save_gif(agent, env_name):
    env = gym.make(env_name,render_mode='rgb_array')
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state = env.reset()[0]
    done = False
    frames=[]
    while not done:
        
        action = agent.act(state)
        frames.append(env.render())
        state, reward, terminate,truncated, _ = env.step(action)

        done = terminate or truncated        
    display_frames_as_gif(frames)
    env.close()