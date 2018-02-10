# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:05:34 2017

@author: dandrews
"""
import sys
sys.path.append('../../nethack_py_interface/nhpyinterface')
from nh_environment import NhEnv
import matplotlib.pyplot as plt
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("matplotlib inline")


from replay_buffer import ReplayBuffer
from actor_critic import ActorCritic
import numpy as np
from keras import backend as K
K.clear_session()
K.set_learning_phase(1)
from collections import namedtuple

class DDPG(object):
    buffer_size =               2048
    batch_size =                1024
    game_episodes_per_update =  512
    epochs = 20000
    grid_size = (20,80)
    input_shape = (21,80,3)
    TAU = 0.1
    min_epsilon = 0.05
    max_epsilon = 0.95
    epsilon_decay = 0.99
    reward_lambda = 0.9
    priority_replay = False
    priortize_low_scores = False
    train_actor=False
    actor_loops = 1
    use_hra = True
    curriculum = None

    def __init__(self):
        self.run_epochs = 0
        self.epochs_total = 0
        self.hybrid_loss_cumulative = []
        self.critic_loss_cumulative = []
        self.critic_target_loss_cumulative = []
        self.actor_loss_cumulative = []
        self.scores_cumulative = []
        self.critic_scores_cumulative = []
        self.actor_scores_cumulative = []
        self.winratio_cumulative = []
        self.epsilon_cumulative = []
        self.epsilon = 0.9
        self.last_lr_change = 0
        e = NhEnv()
        self.environment = e
        self.action_count = e.action_space.n
        self.strategy_count = len(e.strategies)
        self.action_shape = (self.action_count,)
        self.buffer = ReplayBuffer(self.buffer_size)
        num_rewards = len(e.auxiliary_features())
        self.actor_critic = ActorCritic(self.input_shape, self.action_shape, num_rewards)
        self.actor_critic_target = ActorCritic(self.input_shape, self.action_shape, num_rewards)
        self.possible_actions = np.eye(e.action_space.n)[np.arange(e.action_space.n)]


    def train(self, epochs_per_plot=20):
        self.epochs_total = self.epochs + self.run_epochs
        for i in range(self.epochs):
            self.add_replays_to_buffer()
            critic_loss, actor_loss, hybrid_loss = [],[],[]

            # Approximately sample entire replay buffer
            iterations = self.buffer_size // self.batch_size
            for _ in range(iterations):
                buffer = self.buffer.sample_batch(self.batch_size)
                c_loss = self.train_critic_from_buffer(buffer)
                if self.use_hra:
                    h_loss = self.train_hybrid_from_buffer(buffer)
                    hybrid_loss.extend(h_loss)

                critic_loss.extend(c_loss)

                if self.train_actor:
                    a_loss = self.train_actor_from_buffer(buffer)
                    actor_loss.append(a_loss)

            self.actor_critic.target_train(self.actor_critic_target)

            if self.train_actor:
                a_scores = self.run_sample_games(100, use_critic=False)
                self.actor_scores_cumulative.extend(a_scores)
                self.actor_loss_cumulative.append(np.mean(actor_loss))

            # Test agent in 100 games
            c_scores = self.run_sample_games(100, use_critic=True)
            self.critic_scores_cumulative.extend(c_scores)
            self.critic_loss_cumulative.append(np.mean(critic_loss))
            self.hybrid_loss_cumulative.append(np.mean(hybrid_loss))

            # Calculate win/loss ration
            loss = (len(c_scores[c_scores <= 0]))
            win = len(c_scores[c_scores > 0])
            winratio = win / (loss+win+1e-10)
            self.winratio_cumulative.append(winratio)

            self.epsilon_cumulative.append(self.epsilon)
            self.epsilon = 0.99 - winratio # If set to 1 the agent will never play

            self.run_epochs += 1
            if self.run_epochs % epochs_per_plot == 0:
                self.plot_data("Epoch {}/{} of this run".format(i, self.epochs))
            print (self.run_epochs, end=", ")

            if self.is_solved():
                self.plot_data("Done {}".format(i))
                print("\n*********game solved at epoch {}************".format(self.run_epochs))
                break


    def play_one_session(self, random_data=False, use_critic=False):
        e = self.environment
        e.reset()
        moves = []

        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        if self.epsilon > self.max_epsilon:
            self.epsilon = self.max_epsilon

        # In case the agent hasn't had any plays yet get it one for sure
        if np.isnan(self.epsilon):
            self.epsilon = 0.9
            agent_play = True
        elif np.random.rand() > self.epsilon:
            agent_play = True
        else:
            agent_play = False

        # Rollouts were 100% agent or 100% random
        # For larger grids test mixed games
        cumulative_score = 0

        while not e.is_done:
            s = e.data()

            if agent_play:
                a = self.get_action()
                action = np.argmax(a)
            else:
                # replace the agents action at random epsilon% of the time
                action = np.random.randint(self.action_count)
                a = self.possible_actions[action]

            #print("step")
            s_, r, t, info = e.step(action)
            cumulative_score += r
            h = e.auxiliary_features()
            move = namedtuple('move', ['s','a','r', 't','s_','h'])
            (move.s, move.a, move.s_, move.t, move.h) = s, a, s_, t, h
            moves.append(move)

        moves.reverse()
        r = cumulative_score
        for move in moves:
            move.r = r
            r *= self.reward_lambda

        moves.reverse()
        return moves, cumulative_score


    def add_replays_to_buffer(self):
        """
        Fills an empty buffer or adds one batch to existing buffer
        """
        rewards = []
        num = 0
        while num < self.game_episodes_per_update or self.buffer_size > self.buffer.count:
            scored_moves, reward = self.play_one_session()
            rewards.append(reward)
            for move in scored_moves:
                self.buffer.add(move.s, move.a, [move.r], [move.t], move.s_, move.h)
#            if num % 1000 == 0 and num > self.game_episodes_per_update:
            num += len(scored_moves)
#        print("Buffer status {}/{}".format(self.buffer.count, self.buffer_size))

        if self.priority_replay or self.priortize_low_scores:
            s,a,r,t,s_,h = self.buffer.to_batches()
            r = r.squeeze()
            priorities = np.zeros_like(r)
            # Adjust priorities by unpexpected Q and/or low scores
            if self.priority_replay:
                q = self.actor_critic_target.critic.predict([s,a]).squeeze()
                #q = self.actor_critic.critic.predict([s,a]).squeeze()
                priorities = np.abs(q-r)
            if self.priortize_low_scores:
                    priorities -= r
            self.buffer.set_sample_weights(-priorities)
        return rewards

    def train_critic_from_buffer(self, buffer):
        s_batch, a_batch, r_batch, t_batch, s2_batch, h_batch = buffer
        loss = self.actor_critic.train_critic(s_batch, a_batch, r_batch)
        return [loss]

    def train_hybrid_from_buffer(self, buffer):
        s_batch, a_batch, r_batch, t_batch, s2_batch, h_batch = buffer
        loss = self.actor_critic.train_hybrid(s_batch, a_batch, h_batch)
        return [loss]

    def train_actor_from_buffer(self, buffer: ReplayBuffer):
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer
        loss = self.actor_critic.train_actor(s_batch, a_batch)
        return loss

    def is_solved(self):
        # todo define "solved" criteria for initial tests
        return False


    def plot_data(self, title = ""):
        ipython.magic("matplotlib inline")
        title_header = """
Input: {}, Prioritize Bad Q {}, Prioritize Score: {}
Buffer Size: {}, Batch Size: {}, rpe: {}
Use Hybrid Rewards: {} Curriculum: {}""".format(
        self.grid_size,
        self.priority_replay,
        self.priortize_low_scores,
        self.buffer_size,
        self.batch_size,
        self.game_episodes_per_update,
        self.use_hra,
        self.environment.curriculum
        )

        title = title + title_header

        fig, ax = plt.subplots(2,2, figsize=(10, 10))
        ax1 = ax[0,0]
        ax2 = ax[0,1]
        ax3 = ax[1,0]
        ax4 = ax[1,1]
        fig.suptitle(title)

        ax1.set_ylim(ymax=1.1, ymin=0)
        ax1.plot(self.epsilon_cumulative, 'r', label="Epsilon")
        ax1.plot(self.winratio_cumulative, label='moving avg win ratio')
        ax1.legend()

        smoothing = (len(self.critic_scores_cumulative) // 100 )+1
        ax2.axhline(self.benchmark, color='r', label="Solve Score")
        ax2.axhline(0.0, label="0.0")
        if self.train_actor:
            ax2.plot(self.running_mean(self.actor_scores_cumulative,smoothing), label='agent scores')
        ax2.plot(self.running_mean(self.critic_scores_cumulative,smoothing), color='orange', label='critic scores')
        ax2.legend()

        ax3.set_yscale('log')
        ax3.axhline(0, color='r')
        ax3.axhline(1, color='r', label="1.0")
        ax3.plot(self.critic_loss_cumulative, label="critic loss")
        ax3.legend()

        ax4.set_yscale('log')
        ax4.axhline(0, color='r')
        ax4.axhline(1, color='b', label='1.0')
        if self.train_actor:
            ax4.plot(self.actor_loss_cumulative, label="actor metric")
        if self.use_hra:
            ax4.plot(self.hybrid_loss_cumulative, color='green', label="hybrid loss")
        ax4.legend()

        plt.show()

    def get_action(self):
        state = np.array(self.environment.data())
        state = np.expand_dims(state, axis=0)
        if self.train_actor:
            action = self.actor_critic_target.actor.predict(state)[0]
        else:
            action = self.get_best_action_by_q()
        return action

    def get_best_action_by_q(self):
        s = self.environment.data()
        s1 = np.expand_dims(s, axis=0)
        s4 = np.repeat(s1, self.action_shape[0], axis=0)
        pred = self.actor_critic_target.critic.predict([s4,self.possible_actions])
        return self.possible_actions[np.argmax(pred)]

    def run_sample_games(self, num_games = 100, egreedy=True, use_critic=False, stop_on_loss=False):
        scores = []
        e = self.environment
        for i in range(num_games):
            s = e.reset()
            while not e.is_done:
                if use_critic:
                    choice = np.argmax(self.get_best_action_by_q())
                else:
                    s1 = s.reshape(((1,) + s.shape))
                    pred = self.actor_critic.actor.predict(s1)[0]
                    if egreedy:
                        choice = np.argmax(pred)
                    else:
                        choice = np.random.choice(len(pred), p = pred)
                s, r, done, info = e.step(choice)
            scores.append(r)
            if r < 0 and stop_on_loss:
                print("loss detected")
                break
        return np.array(scores)

    def running_mean(self, x, N: int):
        N = int(N)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def softmax(self, a):
        a -= np.min(a)
        a = np.exp(a)
        a /= np.sum(a)
        return a


    def agent_play(self,
                   title="",
                   egreedy=True,
                   random_agent=False,
                   save=False,
                   use_critic=True,
                   frame_pause=0.35):
        from IPython import get_ipython
        ipython = get_ipython()

        e = self.environment
        s = e.reset()
        self.start_state = e.data()
        ann = None
        index = 0
        plt.close()

        figManager = plt.get_current_fig_manager()
        if not 'qt5' in str(figManager):
            ipython.magic("matplotlib qt5")

        while True:
            annotations = self.show_turn(title, index, egreedy, save)
            plt.show()
            #If not inline, bring to front.
            if index == 0:
                plt.pause(1e-9)
                fig = plt.gcf()
                fig.canvas.manager.window.showMinimized()
                fig.canvas.manager.window.showNormal()
                plt.pause(frame_pause)
            plt.pause(frame_pause)

            index += 1
            for ann in annotations:
                ann.remove()
            s1 = s.reshape(((1,) + s.shape))

            if use_critic:
                s2 = np.repeat([e.data()], self.action_shape[0], axis=0)
                pred = self.actor_critic.critic.predict([s2, self.possible_actions]).squeeze()
            else:
                pred = self.actor_critic.actor.predict(s1).squeeze()

            pred = self.softmax(pred)

            if egreedy:
                choice = np.argmax(pred)
            else:
                choice = np.random.choice(len(pred), p = pred)

            if random_agent:
                choice = np.random.choice(len(pred))

            s, r, done, info = e.step(choice)

            if e.is_done:
                ann = self.show_turn(title, index, egreedy, save)
                plt.show()
                plt.pause(frame_pause * 2)
                break
        ipython.magic("matplotlib inline")
        return e.cumulative_score, e.found_exit

    def show_turn(self, title, index, egreedy, save):
        plt.title(title)
        plt.imshow(self.e.data())


#%%


###################################################
if __name__ == '__main__':
###################################################

#%%

    np.set_printoptions(suppress=True)
    nhdqn = DDPG()
    try:
        nhdqn.train()
    except:
        nhdqn.environment.nhc.close()
        raise


