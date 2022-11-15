# coding=utf-8
import torch as th

from .agents import get_agent


class Population(object):

    def __init__(self,
                 n_agents : int,
                 agent_names : list,
                 sender_names : list ,
                 untrained_sender_names: list,
                 untrained_receiver_names: list,
                 receiver_names : list,
                 agent_repertory : dict,
                 game_params : dict,
                 pairs_prob : th.Tensor = None,
                 imitation_probs : th.Tensor = None,
                 device : str = "cpu") -> None:

        """
        :param n_agents: number of agents (int 0)
        :param list_agents: list of agents (list [])
        :param pairs_prob: vertices probabilities (float [n_agents,n_agents])
        """

        # Agents
        self.n_agents = n_agents
        self.agent_names = agent_names
        self.sender_names = sender_names
        self.receiver_names = receiver_names
        self.untrained_sender_names = untrained_sender_names
        self.untrained_receiver_names = untrained_receiver_names
        self.agents = {}
        for agent_name in self.agent_names:

            agent = get_agent(agent_name=agent_name,
                              agent_repertory=agent_repertory,
                              game_params=game_params,
                              device=device)

            self.agents[agent_name] = agent


        # Communication graph
        self.pairs_prob = pairs_prob/pairs_prob.sum()
        self.imitation_probs = imitation_probs/imitation_probs.sum()

    def save_models(self,
                    save_dir:str="/models",
                    add_info:str=""):

        for agent_name in self.agents:
            agent = self.agents[agent_name]
            if agent.sender is not None:
                th.save(agent.sender.state_dict(), f"{save_dir}/{agent_name}_sender_{add_info}.pt")
            if agent.object_encoder is not None:
                th.save(agent.object_encoder.state_dict(), f"{save_dir}/{agent_name}_object_encoder_{add_info}.pt")
            if agent.receiver is not None:
                th.save(agent.receiver.state_dict(), f"{save_dir}/{agent_name}_receiver_{add_info}.pt")
            if agent.object_decoder is not None:
                th.save(agent.object_decoder.state_dict(), f"{save_dir}/{agent_name}_object_decoder_{add_info}.pt")
            if agent.object_projector is not None:
                th.save(agent.object_projector.state_dict(), f"{save_dir}/{agent_name}_object_projector_{add_info}.pt")

class FullyConnectedPopulation(Population):

    def __init__(self,n_agents : int,
                 agent_names : list,
                 agent_repertory: dict,
                 game_params : dict,
                 device : str = "cpu") -> None:
        pairs_prob = 1-th.eye(n_agents)  # type: th.Tensor
        sender_names, receiver_names = [], []
        super().__init__(n_agents = n_agents,
                         agent_names = agent_names,
                         sender_names=sender_names,
                         receiver_names=receiver_names,
                         agent_repertory = agent_repertory,
                         game_params = game_params,
                         pairs_prob = pairs_prob,
                         device = device)

class UnidirectionalFullyConnectedPopulation(Population):

    def __init__(self,
                 n_agents : int,
                 agent_names : list,
                 agent_repertory : dict,
                 game_params : dict,
                 population_graph : th.Tensor,
                 is_sender : list,
                 is_receiver : list,
                 is_imitator : list,
                 is_trained: list,
                 device : str = "cpu"
                 ) -> None:

        if population_graph is None:
            # Agents are not talking to themselves
            population_graph = 1-th.eye(n_agents)
            imitation_probs = th.zeros(n_agents)

            # Ensure that senders do not receive messages and receivers do not send message
            for i in range(n_agents):
                if not is_sender[i] or not is_trained[i]:
                    population_graph[i,:]*=0
                if not is_receiver[i] or not is_trained[i]:
                    population_graph[:,i]*=0
                if is_imitator[i]:
                    imitation_probs[i]=1
        else:
            population_graph = th.Tensor(population_graph)


        sender_names, receiver_names, untrained_sender_names, untrained_receiver_names = [], [], [], []
        imitator_names = []
        for i in range(n_agents):
            if is_sender[i]:
                if is_trained[i]:
                    sender_names.append(agent_names[i])
                else:
                    untrained_sender_names.append(agent_names[i])
            if is_receiver[i]:
                if is_trained[i]:
                    receiver_names.append(agent_names[i])
                else:
                    untrained_receiver_names.append(agent_names[i])
            if is_imitator[i]:
                imitator_names.append(agent_names[i])


        super().__init__(n_agents = n_agents,
                         agent_names = agent_names,
                         sender_names = sender_names,
                         receiver_names = receiver_names,
                         untrained_sender_names=untrained_sender_names,
                         untrained_receiver_names=untrained_receiver_names,
                         game_params = game_params,
                         agent_repertory = agent_repertory,
                         pairs_prob = population_graph,
                         imitation_probs=imitation_probs,
                         device = device)


def build_population(population_params : dict,
                     agent_repertory : dict,
                     game_params : dict,
                     device : str = "cpu") -> Population:

    """

    :param n_agents:
    :param list_agents:
    :param population_type:
    :param print_info_population:
    :return population @type Population
    """

    n_agents = population_params["n_agents"]
    agent_names = population_params["list_agents"]
    population_type = population_params["population_type"]
    communication_graph = population_params["communication_graph"]

    if "population_graph" in population_params:
        population_graph = population_params["population_graph"]
    else:
        population_graph = None

    if population_type =="Unidirectional":
        is_sender = population_params["is_sender"]
        is_receiver = population_params["is_receiver"]

    if "is_trained" in population_params:
        is_trained = population_params["is_trained"]
    else:
        is_trained = [1]*n_agents

    if "is_imitator" in population_params:
        is_imitator = population_params["is_imitator"]
    else:
        is_imitator = [0]*n_agents

    assert n_agents > 0 and len(agent_names) > 0, "Population should have population size > 0"
    assert n_agents==len(agent_names), "Population size should equal length of agent names"
    assert len(is_sender)==len(is_receiver)==n_agents, "is_sender should be equal to is_receiver = n_agents"

    if population_type=="Unidirectional" and communication_graph == "fully_connected":

        population = UnidirectionalFullyConnectedPopulation(n_agents = n_agents,
                                                            agent_names = agent_names,
                                                            agent_repertory = agent_repertory,
                                                            population_graph=population_graph,
                                                            game_params = game_params,
                                                            is_sender = is_sender,
                                                            is_receiver = is_receiver,
                                                            is_trained = is_trained,
                                                            is_imitator=is_imitator,
                                                            device=device)
    else:
        population = FullyConnectedPopulation(n_agents = n_agents,
                                              agent_names = agent_names,
                                              agent_repertory = agent_repertory,
                                              game_params = game_params,
                                              device = device)

    return population