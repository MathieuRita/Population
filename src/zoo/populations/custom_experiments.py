from .utils import generate_json
import argparse

def get_params(params):

    parser = argparse.ArgumentParser()

    # json
    parser.add_argument('--default_population_json', type=str, help="Path with list of involved agents")
    parser.add_argument('--default_agents_json', type=str, help="Path to agents list")
    parser.add_argument('--default_game_json', type=str, help="Path to agents list")
    parser.add_argument('--default_training_json', type=str, help="Path to training parameters")

    # Directory for tensorboard logs
    parser.add_argument('--experiments_dir', type=str, default = "", help="Directory where experiments info are saved")
    parser.add_argument('--logs_dir', type=str, default="", help="Directory logs are saved")
    parser.add_argument('--base_experiment_name', type=str, default="", help="Main name for the experiments")

    args = parser.parse_args(params)

    return args

def prepare_experiments(params):

    opts = get_params(params)

    # Sweep over p_steps

    list_p_step_sender = [1,1,1,1,1,1,1,0.5,0.2,0.1,0.01,0.001,0.0000001]
    list_p_step_receiver = [1,0.5,0.2,0.1,0.01,0.001,0.0000001,1,1,1,1,1,1]

    assert len(list_p_step_sender)==len(list_p_step_receiver), "list_p_sender does not same size as list_p_receiver"

    n_expe = len(list_p_step_sender)

    for expe_i in range(n_expe):

        p_step_sender = list_p_step_sender[expe_i]
        p_step_receiver = list_p_step_receiver[expe_i]

        experiment_name = f"{opts.base_experiment_name}_{p_step_sender}_{p_step_receiver}"

        new_agents = list()
        new_agents.append({"new_name":"sender_A",
                           "default_name":"sender_default_A",
                           "param_changes":{"sender_optim_params":{"optimizer":"adam",
                                                                    "loss":"REINFORCE",
                                                                    "reward":"log",
                                                                    "baseline":"normalization_batch",
                                                                    "p_step":p_step_sender,
                                                                    "lr":0.0005,
                                                                    "entropy_reg_coef":0.005
                                                                    }}})
        new_agents.append({"new_name": "sender_A_0",
                           "default_name": "sender_default_A"})

        new_agents.append({"new_name": "receiver_B",
                           "default_name": "receiver_default_B",
                           "param_changes":{"receiver_optim_params":{"optimizer":"adam",
                                                                     "loss":"cross_entropy",
                                                                     "p_step":p_step_receiver,
                                                                     "lr":0.0005
                                                                }}})
        new_agents.append({"new_name": "receiver_B_0",
                           "default_name": "receiver_default_B"})
        new_agents.append({"new_name": "receiver_A_0",
                           "default_name": "receiver_default_A"})
        new_agents.append({"new_name": "sender_B_0",
                           "default_name": "sender_default_B"})


        new_population = {"param_changes":{"list_agents": ["sender_A","receiver_B","sender_A_0",
                                                           "receiver_A_0","sender_B_0","receiver_B_0"],
                                           "is_trained": [1,1,0,0,0,0],
                                           "is_sender": [1,0,1,0,1,0],
                                           "is_receiver":[0,1,0,1,0,1]}}


        generate_json(experiments_dir=opts.experiments_dir,
                      logs_dir = opts.logs_dir,
                      experiment_name = experiment_name,
                      default_agents_json=opts.default_agents_json,
                      default_population_json=opts.default_population_json,
                      default_game_json=opts.default_game_json,
                      default_training_json=opts.default_training_json,
                      new_agents = new_agents,
                      new_population=new_population)

if __name__ == "__main__":
    import sys
    prepare_experiments(sys.argv[1:])
