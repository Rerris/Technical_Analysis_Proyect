from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium.wrappers import AtariPreprocessing, FrameStack


class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, output_shape, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def call(self, inputs):
        return tf.keras.ops.transpose(inputs, [0, 2, 3, 1])


class DQAgent:

    def __init__(self, env_name: str, max_iters: int = 10, max_steps: int = 10_000,
                 gamma: float = 0.9, epsilon: float = 1, epsilon_min: float = 0.1, epsilon_max: float = 1,
                 batch_size: int = 32, learning_rate: float = 0.00025, history_len: int = 100_000, **kwargs) -> None:
        self.env = gym.make(env_name, **kwargs)
        self.action_space = self.env.action_space.n

        self.max_iters = max_iters
        self.max_steps = max_steps

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_range = (epsilon_max - epsilon_min)
        self.batch_size = batch_size

        self.q_network = self.init_q_network()
        self.q_target_network = self.init_q_network()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.loss = tf.keras.losses.Huber()
        self.replay_buffer = deque(maxlen=history_len)
        self.epoch_reward_history = []

    def init_q_network(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Input((4, 84, 84)),
            TransposeLayer(output_shape=(4, 84, 84)),
            # tf.keras.layers.Lambda(lambda tensor: tf.keras.ops.transpose(tensor, [0, 2, 3, 1]),
            #                        output_shape=(84, 84, 4), input_shape=(4, 84, 84)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(self.action_space, activation="linear"),
        ])

    def train(self):
        running_reward = 0
        episode_count = 0
        frame_count = 0

        self.epsilon = self.epsilon_min  # Delete

        # Number of frames to take random action and observe output
        epsilon_random_frames = 1_000  # 50_000
        # Number of frames for exploration
        epsilon_greedy_frames = 10_000  # 1_000_000
        # Train the model after 4 actions
        update_after_actions = 4
        # How often to update the target network
        update_target_network = 10_000

        while True:
            observation, _ = self.env.reset()
            state = np.array(observation)
            episode_reward = 0

            for timestep in range(1, self.max_steps):
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(self.action_space)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = tf.keras.ops.convert_to_tensor(state)
                    state_tensor = tf.keras.ops.expand_dims(state_tensor, 0)
                    action_probs = self.q_network(state_tensor, training=False)
                    # Take best action
                    action = tf.keras.ops.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                self.epsilon -= self.epsilon_range / epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, _, _ = self.env.step(action)
                state_next = np.array(state_next)

                episode_reward += reward

                # Save actions and states in replay buffer
                self.replay_buffer.append((state, action, reward, state_next, done))
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % update_after_actions == 0 and len(self.replay_buffer) > self.batch_size:
                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(self.replay_buffer)), size=self.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([self.replay_buffer[i][0] for i in indices])
                    action_sample = [self.replay_buffer[i][1] for i in indices]
                    rewards_sample = [self.replay_buffer[i][2] for i in indices]
                    state_next_sample = np.array([self.replay_buffer[i][3] for i in indices])
                    done_sample = tf.keras.ops.convert_to_tensor(
                        [float(self.replay_buffer[i][4]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.q_target_network.predict(state_next_sample, verbose=False)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.gamma * tf.keras.ops.amax(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask, so we only calculate loss on the updated Q-values
                    masks = tf.keras.ops.one_hot(action_sample, self.action_space)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = self.q_network(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.keras.ops.sum(tf.keras.ops.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = self.loss(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.q_network.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

                if frame_count % update_target_network == 0:
                    # update the target network with new weights
                    self.q_target_network.set_weights(self.q_network.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                    # Limit the state and reward history

                if done:
                    break

            # Update running reward to check condition for solving
            self.epoch_reward_history.append(episode_reward)
            if len(self.epoch_reward_history) > 100:
                del self.epoch_reward_history[:1]
            running_reward = np.mean(self.epoch_reward_history)

            episode_count += 1
            print(f"Finished episode {episode_count} out of {self.max_iters}")
            if running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break

            if 0 < self.max_iters <= episode_count:  # Maximum number of episodes reached
                print("Stopped at episode {}!".format(episode_count))
                break

        self.q_network.save("Breakout_Q_network.keras")

    def load_model(self):
        print("Loading model...")
        self.q_network = tf.keras.models.load_model("Breakout_Q_network.keras", safe_mode=False,
                                                    custom_objects={'TransposeLayer': TransposeLayer})
        self.q_target_network.set_weights(self.q_network.get_weights())
        print("Successfully loaded model!")

    def run_single_game(self):
        print("Starting game...")
        observation, info = self.env.reset()
        terminated = False
        r = 0
        # self.env.step(1)
        time_steps = 0
        while not terminated:
            state = np.array(observation)
            state_tensor = tf.keras.ops.convert_to_tensor(state)
            state_tensor = tf.keras.ops.expand_dims(state_tensor, 0)
            action_probs = self.q_network(state_tensor, training=False)
            # Take best action
            action = tf.keras.ops.argmax(action_probs[0]).numpy()
            observation, reward, terminated, _, info = self.env.step(action)
            r += reward

            if time_steps == self.max_steps:
                break

            time_steps += 1

        return r


if __name__ == "__main__":
    tf.keras.config.enable_unsafe_deserialization()
    agent = DQAgent("BreakoutNoFrameskip-v4", max_iters=100, render_mode="human")  # , render_mode="human")
    agent.env = AtariPreprocessing(agent.env)
    agent.env = FrameStack(agent.env, 4)

    agent.load_model()
    # agent.train()
    agent.run_single_game()
    # agent.env.close()
