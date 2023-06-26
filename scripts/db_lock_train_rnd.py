import time
import datetime
import torch
import torch_ac
import numpy as np
import sys
import utils
from db_lock_models import ACModel, ACModelShared
from db_lock_models import RandomFeatureNetwork
from envs.diabolical_combination_locks import HighDimensionalDiabolicalCombinationLock
from agents.ppo_rnd import PPOWithRND


import wandb
project_name = 'db_lock_analysis'

config_defaults = dict(
    algo='ppo_rnd',
    env='HighDimensionalDiabolicalCombinationLock',
    env_horizon=100,
    env_obs_noise_sigma=np.sqrt(0.01),
    seed=124,
    log_interval=1,
    save_interval=0,
    procs=16,
    frames=2e7,
    ac_hidden_dim_size=256,
    lr_anneal_frames=1e8,
    frames_per_proc=100,
    batch_size=200,
    epochs=5,
    clip_eps=0.2,
    discount=0.99,
    lr=0.0005,
    gae_lambda=0.95,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    max_grad_norm=2.5,
    optim_eps=1e-8,
    optim_alpha=0.99,
    text=False,
    intrinsic_coef=0.5,
    lr_predictor_factor=0.25,
    rnd_embedding_size=128,
    rnd_hidden_dim_size=128,
    extra_layer=False,
    recurrence=1,
    # --------------------------
    predictor_hidden_dim_size=128,
    shared_ac_architecture=True,
    record_lock_completion=True,
    # evaluation ---------------------------
    eval_every_interval=200,  # zero means no evaluation
    eval_episodes=10,
    greedy_eval=True,
)


def main():
    wandb.init(config=config_defaults, project=project_name)
    args = wandb.config
    args.mem = (args.recurrence > 1) and (args.algo == 'ppo_rvfd_ensemble') and (args.predictor_conditioning == 'history')
    args.recurrent_predictor = False
    # Set run dir

    hash_config = str(hash(frozenset(args.items())))

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}_{hash_config}"

    model_name = default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []

    for i in range(args.procs):
        if args.env == "HighDimensionalDiabolicalCombinationLock":
            envs.append(HighDimensionalDiabolicalCombinationLock(horizon=args.env_horizon,
                                                                 noise_sigma=args.env_obs_noise_sigma,
                                                                 seed=args.seed,
                                                                 noise_seed=args.seed + i + 1))
    for e in envs:
        o = e.reset()
        print(e.row, e.column)
        print(e.optimal_actions)
    txt_logger.info("Environments loaded\n")
    # Load training status

    if args.eval_every_interval > 0:
        if args.env == "HighDimensionalDiabolicalCombinationLock":
            eval_env = HighDimensionalDiabolicalCombinationLock(horizon=args.env_horizon,
                                                                 noise_sigma=args.env_obs_noise_sigma,
                                                                 seed=args.seed,
                                                                 noise_seed=args.seed + args.procs + 100)

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_lock_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    if args.shared_ac_architecture:
        acmodel = ACModelShared(obs_space, envs[0].action_space,
                                args.ac_hidden_dim_size, args.mem,
                                args.text)

    else:
        acmodel = ACModel(obs_space, envs[0].action_space,
                          args.ac_hidden_dim_size)

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))
    wandb.log({"model details": "{}\n".format(acmodel)})

    random_network = RandomFeatureNetwork(obs_space,
                                          args.rnd_embedding_size,
                                          args.rnd_hidden_dim_size)


    predictor_network = RandomFeatureNetwork(obs_space,
                                                 args.rnd_embedding_size,
                                                 args.predictor_hidden_dim_size,
                                                 extra_layer=args.extra_layer)

    txt_logger.info("{}\n".format(predictor_network))
    # Load algo
    # print(random_network)
    # print(predictor_network)

    if args.algo == "ppo":

        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)

    elif args.algo == "ppo_rnd":

        algo = PPOWithRND(envs=envs, acmodel=acmodel, device=device, num_frames_per_proc=args.frames_per_proc, 
                          discount=args.discount, lr=args.lr, gae_lambda=args.gae_lambda,
                          entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef, 
                          max_grad_norm=args.max_grad_norm, recurrence=args.recurrence,
                          adam_eps=args.optim_eps, clip_eps=args.clip_eps, epochs=args.epochs,
                          batch_size=args.batch_size, preprocess_obss=preprocess_obss, reshape_reward=None,
                          intrinsic_reward_coef=args.intrinsic_coef, record_lock_completion=args.record_lock_completion, 
                          pseudo_reward_generator=random_network, predictor=predictor_network, 
                          lr_predictor=args.lr_predictor_factor * args.lr)


    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            # print(logs.keys())

            if 'rnd' in args.algo:

                header += ["learning_rate"]
                data += [logs["learning_rate"]]

                header += ["learning_rate_predictor"]
                data += [logs["learning_rate_predictor"]]

                header += ["grad_norm_predictor", "predictor_loss"]
                data += [logs["grad_norm_predictor"], logs["predictor_loss"]]

                header += ["mean_intrinsic_reward_batch", "min_intrinsic_reward_batch",
                          "max_intrinsic_reward_batch", "std_intrinsic_reward_batch"]

                data += [logs["mean_intrinsic_reward_batch"], logs["min_intrinsic_reward_batch"],
                        logs["max_intrinsic_reward_batch"], logs["std_intrinsic_reward_batch"]]

                header += ["mean_cumulant_batch", "min_cumulant_batch",
                          "max_cumulant_batch", "std_cumulant_batch"]

                data += [logs["mean_cumulant_batch"], logs["min_cumulant_batch"],
                        logs["max_cumulant_batch"], logs["std_cumulant_batch"]]

                if args.record_lock_completion:
                    header += ["farthest_column_visited"]
                    data += [logs["farthest_column_visited"]]

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            log_dict = {}
            for field, value in zip(header, data):
                log_dict[field] = value

            wandb.log(log_dict)
        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(),
                      "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        if args.eval_every_interval > 0 and update % args.eval_every_interval== 0:

            r_obs, _ = eval_env.reset()
            # print(eval_env.agent_pos)

            # r_memory = torch.zeros(1, acmodel.memory_size,
            #                        device=device)
            # r_data = []
            # r_ep_dict = {}
            r_ep_num = 0
            r_ep_return = 0

            # r_obs_list = []
            # r_action_list = []
            # r_reward_list = []

            # r_frames = []
            eval_returns_list = []

            # Create a window to view the environment
            # env.render('human')

            while r_ep_num < args.eval_episodes:

                # r_frames.append(np.moveaxis(eval_env.render("rgb_array"),
                #                            2, 0))

                # r_obs_list.append(r_obs["image"])
                r_preprocessed_obs = preprocess_obss([r_obs], device=device)

                r_dist, r_value = acmodel(r_preprocessed_obs)

                if args.greedy_eval:
                    r_action = r_dist.probs.max(1, keepdim=True)[1]
                else:
                    r_action = r_dist.sample()

                # r_action_list.append(r_action.cpu().numpy())

                r_obs, r_reward, r_terminated, r_truncated, _ = eval_env.step(r_action.cpu().numpy())

                r_ep_return += r_reward

                # r_reward_list.append(r_reward)

                if r_terminated or r_truncated:

                    # r_ep_dict['features'] = r_obs_list
                    # r_ep_dict['actions'] = np.array(r_action_list).reshape(-1)
                    # r_ep_dict['rewards'] = r_reward_list
                    # r_ep_dict['ep_return'] = r_ep_return
                    eval_returns_list.append(r_ep_return)

                    # r_data.append(r_ep_dict)
                    # print("Episode: ", r_ep_num)
                    # r_ep_dict = {}
                    r_ep_num += 1
                    r_obs, _ = eval_env.reset()
                    # r_memory = torch.zeros(1, acmodel.memory_size,
                    #                       device=device)

                    # r_obs_list = []
                    # r_action_list = []
                    # r_reward_list = []
                    r_ep_return = 0

            mean_eval_return = np.mean(eval_returns_list)
            #print(mean_eval_return)
            wandb.log({"mean_eval_return": mean_eval_return,
                       "max_eval_return": np.max(eval_returns_list),
                       "min_eval_return": np.min(eval_returns_list),
                       "std_eval_return": np.std(eval_returns_list),
                       "frames": num_frames})


if __name__ == '__main__':
    main()
