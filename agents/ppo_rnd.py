import torch
from agents.ppo_with_ir import PPOAlgoWithIR


class PPOWithRND(PPOAlgoWithIR):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, lr_anneal_frames=1e8,
                 preprocess_obss=None, reshape_reward=None, intrinsic_reward_coef=0.1, record_lock_completion=False,
                 pseudo_reward_generator=None, predictor=None, lr_predictor=0.001):

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                         entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                         adam_eps, clip_eps, epochs, batch_size, lr_anneal_frames,
                         preprocess_obss, reshape_reward, intrinsic_reward_coef, record_lock_completion)

        self.pseudo_reward_generator = pseudo_reward_generator
        self.predictor = predictor
        self.lr_predictor = lr_predictor

        self.pseudo_reward_generator.to(self.device)
        self.predictor.to(self.device)

        self.predictor.train()
        self.pseudo_reward_generator.eval()

        # number of pseudo-rewards used
        self.num_cumulants = self.pseudo_reward_generator.output_embedding_size

        self.predictor_criterion = torch.nn.MSELoss()     
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr_predictor, eps=self.adam_eps)

        def lr_lambda(epoch):
            return 1 - min(epoch * self.batch_size/self.epochs, self.lr_anneal_frames) / self.lr_anneal_frames
        self.predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.predictor_optimizer, lr_lambda)
       

    def compute_intrinsic_rewards(self, preprocessed_obs):
        """
        Computes the intrinsic rewards for the given observations using RND
        """
        all_obs = [self.obss[i][j]
                    for i in range(self.num_frames_per_proc)
                    for j in range(self.num_procs)]

        pp_all_obs = self.preprocess_obss(all_obs, device=self.device)

        with torch.no_grad():
            cumulants = self.pseudo_reward_generator(pp_all_obs)
            predicted_values = self.predictor(pp_all_obs)

        self.cumulants = cumulants.reshape(self.num_frames_per_proc, self.num_procs,
                                      self.num_cumulants)
        self.predicted_values = predicted_values.reshape(self.num_frames_per_proc, self.num_procs,
                                      self.num_cumulants)

        intrinsic_rewards = torch.norm(self.cumulants - self.predicted_values, dim=2, p=2)
        return self.intrinsic_reward_coef * intrinsic_rewards

    def update_exps(self):
        exps = super().update_exps()
        exps.targets = self.cumulants.transpose(0, 1).reshape(-1, self.num_cumulants)

        return exps

    def _append_logs_opt(self, logs):
        logs["learning_rate_predictor"] = self.predictor_scheduler.get_last_lr()[0]
        return logs

    def _append_logs_collect(self, logs):
        logs["mean_cumulant_batch"] = self.cumulants.mean().item()
        logs["min_cumulant_batch"] = self.cumulants.min().item()
        logs["max_cumulant_batch"] = self.cumulants.max().item()
        logs["std_cumulant_batch"] = self.cumulants.std().item()
        return logs

    def _get_intrinsic_loss(self, sb):
        """
        Computes the intrinsic loss for the given batch
        """
        predicted_values = self.predictor(sb.obs)
        intrinsic_loss = self.predictor_criterion(predicted_values, sb.targets)
        return intrinsic_loss