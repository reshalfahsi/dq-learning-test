from keras.layers import Dense, Flatten
from keras.models import Sequential
from collections import deque
import gym
import numpy as np
import random
import os
import sys

#global variable
weight_dir = None       # saved weight
env = None              # game environment
D = None                # collection of actions
observetime = None      # number of time steps to observe the game
epsilon = None          # probability doing random move
gamma = None            # discount factor
mb_size = None          # mini batch size
model = None            # network model

def initialization() :
    global env, D, observetime, epsilon, gamma, mb_size, weight_dir
    weight_dir = './dqlearning_weight_test.h5'
    env = gym.make('CartPole-v1')
    D = deque()
    observetime = 500
    epsilon = 0.75
    gamma = 0.90
    mb_size = 400
    pass

def network_model():
    global model, env, weight_dir
    model = Sequential()
    model.add(Dense(20, input_shape=(2,)+env.observation_space.shape,kernel_initializer='uniform',activation='relu'))
    model.add(Flatten())
    model.add(Dense(60,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(60,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(60,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(60,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(env.action_space.n,kernel_initializer='uniform',activation='linear'))
    if os.path.exists(weight_dir) :
        model.load_weights(weight_dir)
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    pass

def run(isTrain=True):
    global model, env, D, observetime, epsilon, gamma, mb_size, weight_dir

    if isTrain:
        #Part 1: Observing!!
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs),axis=1)
        done = False
        action = None
        for t in range(observetime):
            if np.random.rand() <= epsilon:
                action = np.random.randint(0,env.action_space.n, size=1)[0]
            else :
                Q = model.predict(state)
                action = np.argmax(Q)
            observation_new, reward, done, info = env.step(action)
            obs_new = np.expand_dims(observation_new, axis=0)
            state_new = np.append(np.expand_dims(obs_new,axis=0),state[:,:1,:],axis=1)
            D.append((state,action,reward,state_new,done))
            state = state_new
            if done :
                env.reset()
                obs = np.expand_dims(observation,axis=0)
                state = np.stack((obs,obs),axis=1)
        print 'Part 1: Observation!! has been finished'

        #Part 2: Learning from Observation!!
        minibatch = random.sample(D,mb_size)
        input_shape = (mb_size,)+state.shape[1:]
        inputs = np.zeros(input_shape)
        targets = np.zeros((mb_size,env.action_space.n))

        for i in range(0, mb_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]
            done = minibatch[i][4]

            #build bellman equation for the Q function
            inputs[i:i+1] = np.expand_dims(state,axis=0)
            targets[i] = model.predict(state)
            Q_sa = model.predict(state_new)

            if done:
                targets[i,action] = reward
            else:
                targets[i,action] = reward + gamma * np.max(Q_sa)

            model.train_on_batch(inputs, targets)
        print 'Part 2: Learning from Observation!! has been completed'

        #Part 3: Why We don't Play It?!
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs,obs),axis=1)
        done = False
        tot_reward = 0.0
        while not done:
            env.render()
            Q = model.predict(state)
            action = np.argmax(Q)
            observation, reward, done, info = env.step(action)
            obs = np.expand_dims(observation, axis=0)
            state = np.append(np.expand_dims(obs,axis=0), state[:,:1, :],axis=1)
            tot_reward += reward
            if tot_reward >= 400 :
                model.save_weights(weight_dir)
        print 'Part 3: Game ended! Total reward: {r}'.format(r=tot_reward)

    else:
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs,obs),axis=1)
        total_reward = 0.0
        high_score = 0.0

        print 'Let the Game Begin!!'

        while True:
            env.render()
            Q = model.predict(state)
            action = np.argmax(Q)
            observation,reward,done,info = env.step(action)
            obs = np.expand_dims(observation, axis=0)
            state = np.append(np.expand_dims(obs,axis=0), state[:,:1, :],axis=1)
            total_reward += reward
            if done :
                env.reset()
                print ('Reward: {}'.format(total_reward)) + ('' if total_reward<500 else ' HIGH SCORE!! COOL!!')
                if total_reward == 500 :
                    high_score += 1
                    if high_score > 10 :
                        print 'Game Over!! You are Cheater!!'
                        break
                total_reward = 0.0

    print '~@#############@~'
    print ''
    pass



if __name__ == '__main__':
    episode = 250
    isTrain = True
    sum = 0
    for i in sys.argv :
        sum += 1
    if sum > 1:
        if sys.argv[1] == 'False':
            isTrain = False
    initialization()
    network_model()
    for eps in range(episode):
        print '~@#############@~'
        if isTrain:
            print 'Episode: {}'.format(eps)
        run(isTrain)
        if not isTrain :
            break
    pass
