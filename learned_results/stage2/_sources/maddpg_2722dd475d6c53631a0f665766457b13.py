import numpy as np
import tensorflow as tf

from gym import Space
from gym.spaces import Discrete

from tf2marl.agents.AbstractAgent import AbstractAgent
from tf2marl.common.util import space_n_to_shape_n, clip_by_local_norm


class MADDPGAgent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer, num_units, num_lstm_units,
                 gamma, tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 _run=None):
        """
        An object containing critic, actor and training functions for Multi-Agent DDPG.
        """
        self._run = _run

        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic = MADDPGCriticLstmNetwork(num_layer, num_units, num_lstm_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_target = MADDPGCriticLstmNetwork(num_layer, num_units, num_lstm_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        # self.critic = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        # self.critic_target = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        self.policy = MADDPGPolicyLstmNetwork(num_layer, num_units, num_lstm_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                          self.critic, agent_index)
        self.policy_target = MADDPGPolicyLstmNetwork(num_layer, num_units, num_lstm_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                 self.critic, agent_index)
        # self.policy = MADDPGPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
        #                                   self.critic, agent_index)
        # self.policy_target = MADDPGPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
        #                                          self.critic, agent_index)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.batch_size = batch_size
        self.agent_index = agent_index
        self.decay = gamma
        self.tau = tau

    def action(self, obs):
        """
        Get an action from the non-target policy
        """
        return self.policy.get_action(obs[None])[0]

    def target_action(self, obs):
        """
        Get an action from the non-target policy
        """
        return self.policy_target.get_action(obs)

    def preupdate(self):
        pass

    def update_target_networks(self, tau):
        """
        Implements the updates of the target networks, which slowly follow the real network.
        """
        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            # net_weights = np.array(net.get_weights())
            # target_net_weights = np.array(target_net.get_weights())
            # new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            # numpyのwarningが出るので以下に書き換え
            net_weights = net.get_weights()
            target_net_weights = target_net.get_weights()
            new_weights = []
            for net_weight, target_net_weight in zip(net_weights, target_net_weights):
                new_weight = tau * net_weight + (1.0 - tau) * target_net_weight    
                new_weights.append(new_weight)
            
            target_net.set_weights(new_weights)

        update_target_network(self.critic.model, self.critic_target.model)
        update_target_network(self.policy.model, self.policy_target.model)

    def update(self, agents, step):
        """
        Update the agent, by first updating the critic and then the actor.
        Requires the list of the other agents as input, to determine the target actions.
        """
        assert agents[self.agent_index] is self

        if self.prioritized_replay:
            obs_n, acts_n, rew_n, next_obs_n, done_n, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones(rew_n.shape)

        # Train the critic, using the target actions in the target critic network, to determine the
        # training target (i.e. target in MSE loss) for the critic update.
        target_act_next = [a.target_action(obs) for a, obs in zip(agents, next_obs_n)]
        target_q_next = self.critic_target.predict(next_obs_n, target_act_next)
        q_train_target = rew_n[:, None] + self.decay * target_q_next

        td_loss = self.critic.train_step(obs_n, acts_n, q_train_target, weights).numpy()[:, 0]

        # Train the policy.
        policy_loss = self.policy.train(obs_n, acts_n)

        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_loss + self.prioritized_replay_eps)

        # Update target networks.
        self.update_target_networks(self.tau)

        # self._run.log_scalar('agent_{}.train.policy_loss'.format(self.agent_index), policy_loss.numpy(), step)
        # self._run.log_scalar('agent_{}.train.q_loss0'.format(self.agent_index), np.mean(td_loss), step)

        return [td_loss, policy_loss]

    def save(self, fp):
        self.critic.model.save_weights(fp + 'critic.h5',)
        self.critic_target.model.save_weights(fp + 'critic_target.h5')
        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')

    def load(self, fp):
        self.critic.model.load_weights(fp + 'critic.h5')
        self.critic_target.model.load_weights(fp + 'critic_target.h5')
        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')


class MADDPGPolicyNetwork(object):
    def __init__(self, num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.num_layers = num_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.gumbel_temperature = gumbel_temperature
        self.q_network = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=self.obs_n_shape[agent_index])

        self.hidden_layers = []
        for idx in range(num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))

        # connect layers
        x = self.obs_input
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        """
        Produces Gumbel softmax samples from the input log-probabilities (logits).
        These are used, because they are differentiable approximations of the distribution of an argmax.
        """
        uniform_noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(uniform_noise))
        noisy_logits = gumbel + logits  # / temperature
        return tf.math.softmax(noisy_logits)

    def forward_pass(self, obs):
        """
        Performs a simple forward pass through the NN.
        """
        x = obs
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network
        return outputs

    @tf.function
    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        if self.use_gumbel:
            outputs = self.gumbel_softmax_sample(outputs)
        return outputs

    @tf.function
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            x = self.forward_pass(obs_n[self.agent_index])
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x  # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
            else:
                act_n[self.agent_index] = x
            q_value = self.q_network._predict_internal(obs_n + act_n)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss


class MADDPGCriticNetwork(object):
    def __init__(self, num_hidden_layers, units_per_layer, lr, obs_n_shape, act_shape_n, act_type, agent_index):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.num_layers = num_hidden_layers
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape_n = act_shape_n
        self.act_type = act_type

        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(self.act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.hidden_layers = []
        for idx in range(num_hidden_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear',
                                                  name='ag{}crit_out{}'.format(agent_index, idx))

        # connect layers
        x = self.input_concat_layer(self.obs_input_n + self.act_input_n)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,  # list concatenation
                                    outputs=[x])
        self.model.compile(self.optimizer, loss='mse')

    def predict(self, obs_n, act_n):
        """
        Predict the value of the input.
        """
        return self._predict_internal(obs_n + act_n)

    @tf.function
    def _predict_internal(self, concatenated_input):
        """
        Internal function, because concatenation can not be done in tf.function
        """
        x = self.input_concat_layer(concatenated_input)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)
        return x

    def train_step(self, obs_n, act_n, target_q, weights):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        return self._train_step_internal(obs_n + act_n, target_q, weights)

    @tf.function
    def _train_step_internal(self, concatenated_input, target_q, weights):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            x = self.input_concat_layer(concatenated_input)
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            q_pred = self.output_layer(x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss * weights)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return td_loss

# 画像を入力として使用するポリシー
class MADDPGPolicyConvNetwork(object):
    def __init__(self, num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.num_layers = num_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.gumbel_temperature = gumbel_temperature
        self.q_network = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=self.obs_n_shape[agent_index])
        
        self.conv_layers = []
        conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 5, activation='relu')
        self.conv_layers.append(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))
        self.conv_layers.append(pool1)
        conv2 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu')
        self.conv_layers.append(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_layers.append(pool2)
        
        self.hidden_layers = []
        for idx in range(self.num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}_output'.format(agent_index))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}_output'.format(agent_index))
        # # 複数gpu用の処理
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        # connect layers
        x = self.obs_input
        for idx in range(len(self.conv_layers)):
            x = self.conv_layers[idx](x)
        
        x = tf.keras.layers.Flatten()(x)
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        x = self.output_layer(x)
        
        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        """
        Produces Gumbel softmax samples from the input log-probabilities (logits).
        These are used, because they are differentiable approximations of the distribution of an argmax.
        """
        uniform_noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(uniform_noise))
        noisy_logits = gumbel + logits  # / temperature
        return tf.math.softmax(noisy_logits)

    def forward_pass(self, obs):
        """
        Performs a simple forward pass through the NN.
        """
        x = obs
        for idx in range(len(self.conv_layers)):
            x = self.conv_layers[idx](x)
        
        x = tf.keras.layers.Flatten()(x)
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network
        return outputs

    @tf.function
    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        if self.use_gumbel:
            outputs = self.gumbel_softmax_sample(outputs)
        return outputs

    @tf.function
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            x = self.forward_pass(obs_n[self.agent_index])
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x  # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
            else:
                act_n[self.agent_index] = x
            q_value = self.q_network._predict_internal(obs_n, act_n)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss
    

class MADDPGCriticConvNetwork(object):
    def __init__(self, num_hidden_layers, units_per_layer, lr, obs_n_shape, act_shape_n, act_type, agent_index):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.num_layers = num_hidden_layers
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape_n = act_shape_n
        self.act_type = act_type

        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(self.act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.conv_layers = []
        conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 5, activation='relu')
        self.conv_layers.append(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))
        self.conv_layers.append(pool1)
        conv2 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu')
        self.conv_layers.append(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_layers.append(pool2)
        
        self.hidden_layers = []
        for idx in range(self.num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)
        
        self.output_layer = tf.keras.layers.Dense(1, activation='linear',
                                                  name='ag{}crit_out'.format(agent_index))
        
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        # connect layers
        # x = tf.keras.layers.Concatenate(axis=0)(self.obs_input_n)
        x = self.obs_input_n[0]
        for idx in range(len(self.conv_layers)):
            x = self.conv_layers[idx](x)
        
        x = tf.keras.layers.Flatten()(x)
        x = self.input_concat_layer([x] + [self.input_concat_layer(self.act_input_n)])  # ここでactionを結合する
        
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        x = self.output_layer(x)
        
        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,  # list concatenation
                                    outputs=[x])
        self.model.compile(self.optimizer, loss='mse')
        self.model.summary()

    def predict(self, obs_n, act_n):
        """
        Predict the value of the input.
        """
        return self._predict_internal(obs_n, act_n)

    @tf.function
    def _predict_internal(self, obs_n, act_n):
        """
        Internal function, because concatenation can not be done in tf.function
        """
        # x = tf.keras.layers.Concatenate(axis=0)(obs_n)
        x = obs_n[0]
        for idx in range(len(self.conv_layers)):
            x = self.conv_layers[idx](x)
        
        x = tf.keras.layers.Flatten()(x)
        x = self.input_concat_layer([x] + [self.input_concat_layer(act_n)])  # ここでactionを結合する
        
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        x = self.output_layer(x)
        return x

    def train_step(self, obs_n, act_n, target_q, weights):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        return self._train_step_internal(obs_n, act_n, target_q, weights)

    @tf.function
    def _train_step_internal(self, obs_n, act_n, target_q, weights):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            # x = tf.keras.layers.Concatenate(axis=0)(obs_n)
            x = obs_n[0]
            for idx in range(len(self.conv_layers)):
                x = self.conv_layers[idx](x)
            
            x = tf.keras.layers.Flatten()(x)
            x = self.input_concat_layer([x] + [self.input_concat_layer(act_n)])  # ここでactionを結合する
            
            for idx in range(len(self.hidden_layers)):
                x = self.hidden_layers[idx](x)
            
            q_pred = self.output_layer(x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss * weights)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return td_loss

# LSTM層を追加したポリシー
class MADDPGPolicyLstmNetwork(object):
    def __init__(self, num_layers, units_per_layer, num_lstm_units, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.num_layers = num_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.gumbel_temperature = gumbel_temperature
        self.q_network = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.input_concat_layer = tf.keras.layers.Concatenate()
        
        self.max_num_O = 3
        # 1つの障害物に対するobservationの数
        self.Os_obs_num = 4
        # 最初からmaxの分引いたものをshapeとする
        self.obs_main_shape = self.obs_n_shape[agent_index] - self.Os_obs_num * self.max_num_O
        # maxで3(障害物の数)*4(各座標と移動ベクトル)のarrayを想定  
        self.obs_Os_shape = np.array([self.max_num_O, self.Os_obs_num])
        
        ### set up network structure
        self.obs_input_main = tf.keras.layers.Input(shape=self.obs_main_shape)
        self.obs_input_Os = tf.keras.layers.Input(shape=self.obs_Os_shape)
        
        self.masking_layer = tf.keras.layers.Masking(mask_value=-10., input_shape=self.obs_Os_shape)
        self.LSTM_layer = tf.keras.layers.LSTM(num_lstm_units)
        
        self.hidden_layers = []
        for idx in range(self.num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)
            

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}_output'.format(agent_index))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}_output'.format(agent_index))

        # connect layers
        x = self.obs_input_Os
        x = self.masking_layer(x)
        x = self.LSTM_layer(x)
        x = self.input_concat_layer([x] + [self.obs_input_main])
        
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        x = self.output_layer(x)
        
        self.model = tf.keras.Model(inputs=[self.obs_input_main] + [self.obs_input_Os], outputs=[x])

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        """
        Produces Gumbel softmax samples from the input log-probabilities (logits).
        These are used, because they are differentiable approximations of the distribution of an argmax.
        """
        uniform_noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(uniform_noise))
        noisy_logits = gumbel + logits  # / temperature
        return tf.math.softmax(noisy_logits)

    def forward_pass(self, obs):
        """
        Performs a simple forward pass through the NN.
        """
        obs_main = obs[:, :-self.Os_obs_num * self.max_num_O]  # 障害物以外の情報
        obs_Os_flat = obs[:, -self.Os_obs_num * self.max_num_O:]  # 障害物の情報
        obs_Os = tf.reshape(obs_Os_flat, (-1, self.max_num_O, self.Os_obs_num))
                
        x = self.input_concat_layer([obs_Os])
        x = self.masking_layer(x)
        x = self.LSTM_layer(x)
        x = self.input_concat_layer([x] + [obs_main])
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network
        return outputs

    @tf.function
    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        if self.use_gumbel:
            outputs = self.gumbel_softmax_sample(outputs)
        return outputs

    @tf.function
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            x = self.forward_pass(obs_n[self.agent_index])
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x  # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
            else:
                act_n[self.agent_index] = x
            q_value = self.q_network._predict_internal(obs_n, act_n)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss
    

class MADDPGCriticLstmNetwork(object):
    def __init__(self, num_hidden_layers, units_per_layer, num_lstm_units, lr, obs_n_shape, act_shape_n, act_type, agent_index):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.num_layers = num_hidden_layers
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape_n = act_shape_n
        self.act_type = act_type

        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n_main = []
        self.obs_input_n_Os = []
        
        self.max_num_O = 3
        # 1つの障害物に対するobservationの数
        self.Os_obs_num = 4
        self.obs_main_shape = self.obs_shape_n[agent_index] - self.Os_obs_num * self.max_num_O  # 最初からmaxの分引いたものをshapeとする
        self.obs_Os_shape = np.array([self.max_num_O, self.Os_obs_num])  # maxで3(障害物の数)*2(各座標)のarrayを想定
        
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n_main.append(tf.keras.layers.Input(shape=self.obs_main_shape, name='obs_in' + str(idx)))
            self.obs_input_n_Os.append(tf.keras.layers.Input(shape=self.obs_Os_shape, name='obs_in_Os' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(self.act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()
        
        self.masking_layer = tf.keras.layers.Masking(mask_value=-10., input_shape=self.obs_Os_shape)
        self.LSTM_layer = tf.keras.layers.LSTM(num_lstm_units)
        
        self.hidden_layers = []
        for idx in range(self.num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)
        
        self.output_layer = tf.keras.layers.Dense(1, activation='linear',
                                                  name='ag{}crit_out'.format(agent_index))
        
        # connect layers
        # x = self.input_concat_layer([self.obs_input_n_Os])
        x = self.obs_input_n_Os[0]  # globalで等しい情報については重ねない → 効率的に学習できるのでは
        x = self.masking_layer(x)
        x = self.LSTM_layer(x)
        x = self.input_concat_layer([self.input_concat_layer(self.obs_input_n_main)] + [x] 
                                    + [self.input_concat_layer(self.act_input_n)])
        
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        x = self.output_layer(x)
        
        self.model = tf.keras.Model(inputs=self.obs_input_n_main + self.obs_input_n_Os + self.act_input_n,  # list concatenation
                                    outputs=[x])
        self.model.compile(self.optimizer, loss='mse')
        # self.model.summary()

    def predict(self, obs_n, act_n):
        """
        Predict the value of the input.
        """
        return self._predict_internal(obs_n, act_n)

    @tf.function
    def _predict_internal(self, obs_n, act_n):
        """
        Internal function, because concatenation can not be done in tf.function
        """
        obs_main_n = []
        obs_Os_n = []
        for idx, obs in enumerate(obs_n):
            obs_main = obs[:, :-self.Os_obs_num * self.max_num_O]  # 障害物以外の情報
            obs_Os_flat = obs[:, -self.Os_obs_num * self.max_num_O:]  # 障害物の情報
            obs_Os = tf.reshape(obs_Os_flat, (-1, self.max_num_O, self.Os_obs_num))
    
            obs_main_n.append(obs_main)
            obs_Os_n.append(obs_Os)
        
        # x = self.input_concat_layer(obs_Os_n)
        x = obs_Os_n[0]
        x = self.masking_layer(x)
        x = self.LSTM_layer(x)
        x = self.input_concat_layer([self.input_concat_layer(obs_main_n)] + [x] + [self.input_concat_layer(act_n)])  # ここでactionを結合する
        
        for idx in range(len(self.hidden_layers)):
            x = self.hidden_layers[idx](x)
        
        x = self.output_layer(x)
        return x

    def train_step(self, obs_n, act_n, target_q, weights):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        return self._train_step_internal(obs_n, act_n, target_q, weights)

    @tf.function
    def _train_step_internal(self, obs_n, act_n, target_q, weights):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            obs_main_n = []
            obs_Os_n = []
            for idx, obs in enumerate(obs_n):
                obs_main = obs[:, :-self.Os_obs_num * self.max_num_O]  # 障害物以外の情報
                obs_Os_flat = obs[:, -self.Os_obs_num * self.max_num_O:]  # 障害物の情報
                obs_Os = tf.reshape(obs_Os_flat, (-1, self.max_num_O, self.Os_obs_num))
        
                obs_main_n.append(obs_main)
                obs_Os_n.append(obs_Os)
            
            # x = self.input_concat_layer(obs_n)
            x = obs_Os_n[0]
            x = self.masking_layer(x)
            x = self.LSTM_layer(x)
            x = self.input_concat_layer([self.input_concat_layer(obs_main_n)] +
                                        [x] + [self.input_concat_layer(act_n)])  # ここでactionを結合する    
            
            for idx in range(len(self.hidden_layers)):
                x = self.hidden_layers[idx](x)
            
            q_pred = self.output_layer(x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss * weights)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return td_loss