import numpy as np
from tf2marl.multiagent.core import World, Agent, Landmark
from tf2marl.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 6 # 增加智能体的数量
        world.num_agents = num_agents
        num_adversaries = 4 # 增加对抗者的数量
        world.n_adversaries = num_adversaries
        num_landmarks = 5 # 增加地标的数量
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True # 允许智能体之间发生碰撞
            agent.silent = False # 允许智能体之间进行通信
            agent.adversary = True if i < num_adversaries else False # 前四个智能体为对抗者，后两个为被捕食者
            agent.leader = True if i == 0 else False # 第一个对抗者为领导者，可以发送通信信息给其他对抗者
            agent.size = 0.15 if agent.adversary else 0.1 # 对抗者的大小为0.15，被捕食者的大小为0.1
            agent.accel = 3.0 if agent.adversary else 4.0 # 对抗者的加速度缩放系数为3.0，被捕食者的为4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3 # 对抗者的最大速度为1.0，被捕食者的为1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True # 允许智能体与地标发生碰撞
            landmark.movable = False # 地标不可移动
            landmark.size = 0.08 if i < 2 else 0.2 # 前两个地标为食物，大小为0.08，后三个地标为树林，大小为0.2
            landmark.food = True if i < 2 else False # 前两个地标为食物，可以被被捕食者吃掉
            landmark.forest = True if i >= 2 else False # 后三个地标为树林，可以为智能体提供掩护
        # make initial conditions
        self.reset_world(world)
        return world

    # def make_world(self):
    #     world = World()
    #     # set any world properties first
    #     world.dim_c = 2
    #     num_agents = 3
    #     world.num_agents = num_agents
    #     num_adversaries = 1
    #     world.n_adversaries = num_adversaries
    #     num_landmarks = num_agents - 1
    #     # add agents
    #     world.agents = [Agent() for i in range(num_agents)]
    #     for i, agent in enumerate(world.agents):
    #         agent.name = 'agent %d' % i
    #         agent.collide = False
    #         agent.silent = True
    #         agent.adversary = True if i < num_adversaries else False
    #         agent.size = 0.15
    #         # Add new properties for complex multi-domain action scenarios
    #         agent.domain = 'domain %d' % (i % 2)  # Assign agents to different domains
    #         agent.complex_action = True  # Enable complex actions
    #     # add landmarks
    #     world.landmarks = [Landmark() for i in range(num_landmarks)]
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.name = 'landmark %d' % i
    #         landmark.collide = False
    #         landmark.movable = False
    #         landmark.size = 0.08
    #         # Add new properties for complex multi-domain action scenarios
    #         landmark.domain = 'domain %d' % (i % 2)  # Assign landmarks to different domains
    #     # make initial conditions
    #     self.reset_world(world)
    #     return world



    # def make_world(self):
    #     world = World()
    #     # set any world properties first
    #     world.dim_c = 2
    #     num_agents = 3
    #     world.num_agents = num_agents
    #     num_adversaries = 1
    #     world.n_adversaries = num_adversaries
    #     num_landmarks = num_agents - 1
    #     # add agents
    #     world.agents = [Agent() for i in range(num_agents)]
    #     for i, agent in enumerate(world.agents):
    #         agent.name = 'agent %d' % i
    #         agent.collide = False
    #         agent.silent = True
    #         agent.adversary = True if i < num_adversaries else False
    #         agent.size = 0.15
    #     # add landmarks
    #     world.landmarks = [Landmark() for i in range(num_landmarks)]
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.name = 'landmark %d' % i
    #         landmark.collide = False
    #         landmark.movable = False
    #         landmark.size = 0.08
    #     # make initial conditions
    #     self.reset_world(world)
    #     return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            return adv_rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)
