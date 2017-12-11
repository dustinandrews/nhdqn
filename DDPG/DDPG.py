# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:05:34 2017

@author: dandrews
"""
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
sess.run(hello)

import sys
sys.path.append('D:/local/machinelearning/textmap')
from tmap import Map

from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import keras
import numpy as np
from keras import backend as K
K.clear_session()

class DDPG(object):
    buffer_size = 1000
    batch_size = 100
    epochs = 10
    input_shape = (3,3)
    decay = 0.9
    TAU = 0.125
    
    
    def __init__(self):
        e = Map(self.input_shape[0],self.input_shape[1])        
        self.output_shape = e.action_space.n
        self.action_input_shape = (1,)
        
        self.environment = e
        
        self.buffer = ReplayBuffer(self.buffer_size)
        
        actor_network = ActorNetwork()        
        self.actor = actor_network.create_actor_network(
                self.input_shape,
                self.output_shape)
        self.actor_target = actor_network.create_actor_network(
                self.input_shape,
                self.output_shape)
        
        critic_network = CriticNetwork()
        self.critic = critic_network.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.action_input_shape
                )        
        self.critic_target = critic_network.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.action_input_shape
                )
        
        

    def step(self):
        state = np.expand_dims(self.environment.data_normalized(), axis=0)        
        prediction = self.actor.predict([state],1)
        return prediction

    def target_train(self, source: keras.models.Model, target: keras.models.Model):
        """
        Nudges target model towards source values
        """
        tau = self.TAU
        source_weights = np.array(source.get_weights())
        target_weights = np.array(target.get_weights())        
        new_weights = tau * source_weights + (1 - tau) * target_weights
        target.set_weights(new_weights)
            
    
    def fill_replay_buffer(self, random_data=False):
        e = self.environment
        rewards = []
        for i in range(self.buffer_size):
            if e.done:
                e.reset()            
            a = self.get_action(random_data)
            s = e.data_normalized()
            (s_, r, t, info) = e.step(a)            
            self.buffer.add(s, [a], [r], [t], s_)
            rewards.append(r)
        return rewards
                            
    def train_critic_from_buffer(self, buffer):
        loss_record = []
        for i in range(self.buffer_size//self.batch_size):
           s_batch, a_batch, r_batch, t_batch, s2_batch = buffer           
           loss = self.critic.train_on_batch([s_batch, a_batch], r_batch)           
           self.target_train(self.critic, self.critic_target)
           loss_record.append(loss)
        return loss_record
           
    def train_actor_from_buffer(self, buffer):
        loss_record = []
        for i in range(self.buffer_size//self.batch_size):
            s_batch, a_batch, r_batch, t_batch, s2_batch  = buffer
            a_batch.resize((self.batch_size,))           
            a_one_hot = np.eye(self.output_shape)[a_batch]            
            critic_predictions = self.critic_target.predict([s_batch,a_batch])
            gradient  = a_one_hot * critic_predictions
            loss = self.actor.train_on_batch(s_batch, gradient)            
            self.target_train(self.actor, self.actor_target)
            loss_record.append(loss)
        return loss_record
           
    def train(self, use_worst=False):
        random_data = False
        actor_loss,critic_loss,critic_target_loss, actor_target_loss, scores= [],[],[],[],[]       
        last_lr_change = 0        
        for i in range(self.epochs):
            s = self.fill_replay_buffer(random_data=random_data)
            if use_worst:
                buffer = self.buffer.sample_worst_batch(self.batch_size)
            else:
                buffer = self.buffer.sample_batch(self.batch_size)
                
            scores.append(np.mean(s))
            c_loss = self.train_critic_from_buffer(buffer)
            a_loss = self.train_actor_from_buffer(buffer) 
            at_loss = self.get_actor_loss_from_buffer(self.actor_target)
            ct_loss = self.get_loss_from_buffer(self.critic_target)            
            ct_avg = np.ones_like(c_loss) * np.mean(ct_loss.squeeze())             
            critic_loss.extend(c_loss)
            actor_loss.extend(a_loss)
            critic_target_loss.extend(ct_avg)
            actor_target_loss.extend(at_loss)
            random_data = False            
            print("{}/{}".format(i, self.epochs))
            if i - last_lr_change > 200:
                mean_loss = np.mean(critic_loss[-50])
                #print(i, mean_loss, end=", ")
                if np.mean(c_loss) >= mean_loss:
                    lr = K.get_value(self.critic.optimizer.lr)
                    #print("Lowering Learning rate {} by order of magnitude.".format(lr))
                    K.set_value(self.critic.optimizer.lr, lr/10)
                    last_lr_change = i
        return critic_loss, critic_target_loss, actor_target_loss, scores
        
    def get_loss_from_buffer(self, model: keras.models.Model):
        s_batch, a_batch, r_batch, t_batch, s2_batch  = self.buffer.sample_batch(self.batch_size)        
        pred = model.predict([s_batch, a_batch])
        delta = np.square(pred - r_batch)
        return delta
    
    def get_actor_loss_from_buffer(self, model: keras.models.Model):
        s_batch, a_batch, r_batch, t_batch, s2_batch  = self.buffer.sample_batch(self.batch_size)        
        pred = model.predict([s_batch])
        delta = np.square(pred - a_batch)
        return delta
            

    def get_action(self, random_data=False):
        if not random_data:
            state = np.array(self.environment.data_normalized())
            state = np.expand_dims(state, axis=0)
            pred = self.actor_target.predict(state)
            pred = pred.squeeze()
            # e-greedy
            #action = np.argmax(pred)
            
            # weighted random
            action = np.random.choice(len(pred), p = pred)
        else: 
            action = np.random.randint(0, self.output_shape)
        return action
    
    def running_mean(self, x, N: int):
        N = int(N)
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N) 

#%%
if __name__ == '__main__':
#%%    
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True)

#%%
    def smoke_test():
        ddpg = DDPG()
        ddpg.epochs=1
        pred = ddpg.step()
        print(pred)
        ddpg.fill_replay_buffer(random_data=True)        
        s_batch, a_batch, r_batch, t_batch, s2_batch = ddpg.buffer.sample_batch(10)    
        critic_loss, critic_target_loss, actor_target_loss, scores = ddpg.train()        
        running_mean = ddpg.running_mean
        running_mean_window = ddpg.batch_size//10
        cl_rm = running_mean(critic_loss, running_mean_window)
        ct_rm = running_mean(critic_target_loss, running_mean_window)
       
        plt.plot(cl_rm , label="critic_loss")
        plt.plot(ct_rm, label="critic_target_loss")
        #plt.plot(actor_target_loss, label="actor_target_loss")
        plt.legend()
        plt.show()
        plt.plot(scores, label="scores")
        plt.legend()
        plt.show()
    
#%%
   
    def agent_play(ddpg):        
        e = ddpg.environment
        s = e.reset()
        while not e.done:
            e.render()
            s1 = s.reshape(((1,) + s.shape))
            a = ddpg.actor_target.predict(s1)            
            #diagnostic critic
            if(e.moves > 5):
                a_ = np.arange(4).reshape(4,1)
                s_ = np.ones(((4,) + ddpg.input_shape)) * s1
                c = ddpg.critic_target.predict([s_,a_])
                print(np.argmax(a), np.argmax(c), c)
            
            choice = np.argmax(a)
#            print(a)
            s, r, done, info = e.step(choice)
            
        e.render()

#%%        
    def check_for_play_errors(ddpg):
        r = 1
        index = 0;
        while r > 0: 
            e = Map(ddpg.input_shape[0],ddpg.input_shape[1])
            s = e.reset()
            while not e.done:
                print(index)
                index += 1
                s1 = s.reshape(((1,) + s.shape))
                a = ddpg.actor_target.predict(s1)            
                #diagnostic critic
                a_ = np.arange(4).reshape(4,1)
                s_ = np.ones(((4,) + ddpg.input_shape)) * s1
                c = ddpg.critic_target.predict([s_,a_])

                choice = np.argmax(a)
    #            print(a)
                s, r, done, info = e.step(choice)
        #print (e.last_render)
        print (e.ac_actions[e.action_index[choice]])
        e.render()
        print(s)
        print(a)
        print(c)

#%%
    def avg_game_len(ddpg, num_games = 100):
        scores = []
        game_len = []
        for i in range(100):
            #action = np.random.choice(len(pred), p = pred)
            e = Map(ddpg.input_shape[0],ddpg.input_shape[1])
            s = e.reset()
            j = 0
            while not e.done:                
                s1 = s.reshape(((1,) + s.shape))
                pred = ddpg.actor_target.predict(s1)[0]                           
                choice = np.random.choice(len(pred), p = pred)
                s, r, done, info = e.step(choice)
                j += 1
                scores.append(e.cumulative_score)
            game_len.append(j)
        #print(i, np.mean(scores))
        return scores, game_len
#%%    


    ddpg = DDPG()
    ddpg.fill_replay_buffer(random_data=True)  
#%%       
    def performance_over_iterations(ddpg, num):
        
        cl,tcl,atl,sc,gl = [],[],[],[],[]
        
        for i in range(num):            
            critic_loss, critic_target_loss, actor_target_loss, scores = ddpg.train()
            cl.extend(critic_loss)
            tcl.extend(critic_target_loss)
            atl.extend(actor_target_loss)
            
            scores, game_len = avg_game_len(ddpg, num)
            sc.extend(scores)
            gl.extend(game_len)
        
        #critic_loss, critic_targert_loss, actor_target_loss, scores, game_len 
        return cl,tcl,atl,sc,gl

#%%    
    
    cl, ctl, atl, sc, gl = [],[],[],[],[]
#%%
    ddpg.epochs = 10
    critic_loss, critic_targert_loss, actor_target_loss, scores, game_len \
    = performance_over_iterations(ddpg, 10)
    cl.extend(critic_loss)
    ctl.extend(critic_targert_loss)
    atl.extend(actor_target_loss)
    sc.extend(scores)
    gl.extend(game_len)
    
  
    data = {
            'critic_loss': cl,
            'critic_target_loss': ctl,
            'actor_target_loss': atl,
            'scores': sc,
            'game_len': gl
            }
    
    for d in data:
        m = data[d]
        plt.plot(ddpg.running_mean(m, len(m)//10), label=d)
        plt.legend()
        plt.show()

