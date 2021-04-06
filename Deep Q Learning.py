# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from simulator import *

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # using tensorflow 1 because this code is in version 1, version 2 does not allow placeholders etc.
import numpy as np
import random


class DeepQAgent():
    def __init__(self, env):  # setting hyper-parameters and initialize NN model
        # set hyperparameters
        self.max_episodes = 3
        self.max_actions = 99
        self.discount = 0.93
        self.exploration_rate = 1.0
        self.exploration_decay = 1.0 / 20000
        # get envirionment
        self.env = env

        # nn_model parameters
        # self.in_units = env.observation_space.n
        self.in_units = 6  # no of state features
        # self.out_units = env.action_space.n
        self.out_units = 5  # no of actions available
        self.hidden_units = int(10)  # can set variably

        # construct nn model
        self._nn_model(env)

        # save nn model
        self.saver = tf.train.Saver()

    def _nn_model(self, env):  # build nn model - so this is in lieu of policy

        self.a0 = tf.placeholder(tf.float32, shape=[1, self.in_units])  # input layer
        # self.a0 = tf.compat.v1.placeholder(tf.float32, shape=[1, self.in_units]) # if using Tensorflow2, disable eagerexecution and use this
        self.y = tf.placeholder(tf.float32, shape=[1, self.out_units])  # ouput layer

        # from input layer to hidden layer
        self.w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units],
                                       dtype=tf.float32))  # weight- will work in both 1 and 2 versions of tensorflow
        self.b1 = tf.Variable(tf.random_uniform([self.hidden_units], 0, 0.01, dtype=tf.float32))  # bias
        self.a1 = tf.nn.relu(tf.matmul(self.a0, self.w1) + self.b1)  # the ouput of hidden layer

        # from hidden layer to output layer
        self.w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32))  # weight
        self.b2 = tf.Variable(tf.random_uniform([self.out_units], 0, 0.01, dtype=tf.float32))  # bias

        # Q-value and Action
        self.a2 = tf.matmul(self.a1, self.w2) + self.b2  # the predicted_y (Q-value) of five actions
        self.action = tf.argmax(self.a2, 1)  # the agent would take the action which has maximum Q-value
        # so we are taking max action, what about exploration?- will come in train()

        # loss function
        self.loss = tf.reduce_sum(tf.square(self.a2 - self.y))

        # upate model
        self.update_model = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss)

    def train(self):  # training the agent
        # get hyper parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay

        with tf.Session() as sess:  # this way no need to close() session
            sess.run(tf.global_variables_initializer())  # initialize tf variables
            for i in range(max_episodes):
                state = env.reset(1)  # reset the environment per episodes
                for j in range(max_actions):
                    # get action and Q-values of all actions
                    action, pred_Q = sess.run([self.action, self.a2],
                                              feed_dict={self.a0: [state]})  # sets an initial Q value
                    # sess.run will simply run a forward loop

                    # if exploring, then taking a random action instead
                    if np.random.rand() < exploration_rate:
                        action[0] = random.choice(env.action_space)

                        # get nextQ in given next_state
                    next_state, rewards, done, info = env.step(action[0])
                    next_Q = sess.run(self.a2, feed_dict={self.a0: [next_state]})

                    # update
                    update_Q = pred_Q
                    update_Q[0, action[0]] = rewards + discount * np.max(
                        next_Q)  # where is learning rate here?, so  leraning rate is 1

                    sess.run([self.update_model],
                             feed_dict={self.a0: [state], self.y: update_Q})
                    state = next_state

                    # if fall in the hole or arrive to the goal, then this episode is terminated.
                    if done:
                        if exploration_rate > 0.001:
                            exploration_rate -= exploration_decay
                        break
            # save model
            save_path = self.saver.save(sess, "./nn_model.ckpt")

    def test(self):  # testing the agent
        # get hyper-parameters
        max_actions = self.max_actions

        with tf.Session() as sess:
            # restore the model
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph("./nn_model.ckpt.meta")  # restore model
            saver.restore(sess, tf.train.latest_checkpoint('./'))  # restore variable

            # testing result
            state = env.reset(1)
            for j in range(max_actions):
                #env.render()  # show the environments
                # always take optimal action
                action, pred_Q = sess.run([self.action, self.a2], feed_dict={self.a0: [state]})
                # update
                next_state, rewards, done, info = env.step(action[0])
                state = next_state
                if done:
                    env.render()
                    break

    def displayQ():  # show information
        pass


# env = Environment().FrozenLakeNoSlippery() # construct the environment
env = Game()
agent = DeepQAgent(env)  # get agent
print("START TRAINING...")
agent.train()
print("\n\nTEST\n\n")
agent.test()
