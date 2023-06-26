import torch
from agents.ppo_with_ir import PPOAlgoWithIR


class PPOWithRCGVF(PPOAlgoWithIR):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, lr_anneal_frames=1e8,
                 preprocess_obss=None, reshape_reward=None, intrinsic_reward_coef=0.1, record_lock_completion=False,
                 pseudo_reward_generator=None, predictor=None, lr_predictor=0.001, 
                 gvf_discount=0.6, gvf_lambda=0.95, use_std=False):

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

        self.gvf_discount = gvf_discount
        self.gvf_lambda = gvf_lambda
        
        self.use_std = use_std

        self.predictor_criterion = torch.nn.MSELoss()     
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr_predictor, eps=self.adam_eps)

        def lr_lambda(epoch):
            return 1 - min(epoch * self.batch_size/self.epochs, self.lr_anneal_frames) / self.lr_anneal_frames
        self.predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.predictor_optimizer, lr_lambda)
       

    def compute_intrinsic_rewards(self, preprocessed_obs):
        """
        Computes the intrinsic rewards for the given observations using RCGVF
        Shape of predicted values:  (n_heads, num_frames_per_proc, num_procs, num_cumulants)
        """
        all_obs = [self.obss[i][j]
                    for i in range(self.num_frames_per_proc)
                    for j in range(self.num_procs)]

        pp_all_obs = self.preprocess_obss(all_obs, device=self.device)

        with torch.no_grad():
            cumulants = self.pseudo_reward_generator(pp_all_obs)
            predicted_values_mult = self.predictor(pp_all_obs)
            next_pseudo_value_mult = self.predictor(preprocessed_obs)
        
        next_pseudo_value_mult = torch.stack(next_pseudo_value_mult)
        predicted_values_mult = torch.stack(predicted_values_mult)

        self.cumulants = cumulants.reshape(self.num_frames_per_proc, self.num_procs,
                                           self.num_cumulants)

        self.predicted_values_mult = predicted_values_mult.reshape(self.predictor.num_heads, 
                                                                   self.num_frames_per_proc, self.num_procs,
                                                                   self.num_cumulants)

        self.pseudo_returns_mult = self._compute_pseudo_returns(next_pseudo_value_mult)


        #### Computation of the intrinsic rewards
        pred_error = torch.mean(torch.pow((self.pseudo_returns_mult - self.predicted_values_mult), 2), dim=0)
        if self.use_std:
            var_preds = torch.std(self.predicted_values_mult, dim=0)
        else:
            var_preds = torch.var(self.predicted_values_mult, dim=0)
        intrinsic_rewards = torch.sum(var_preds * pred_error, dim=-1)

        return self.intrinsic_reward_coef * intrinsic_rewards

    def _compute_pseudo_returns(self, next_pseudo_value_mult):
        
        pseudo_return_mult = torch.zeros(self.predicted_values_mult.shape,
                                         device=self.device)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_mask = next_mask.unsqueeze(1).repeat(1, self.num_cumulants)  # to account for multiple cumulants
            next_value = (self.gvf_lambda * pseudo_return_mult[:, i+1] + (1 - self.gvf_lambda) * self.predicted_values_mult[:, i+1]) if i < self.num_frames_per_proc - 1 else next_pseudo_value_mult


            pseudo_return_mult[:, i, :, :] = self.cumulants[i] + self.gvf_discount * next_mask * next_value

        return pseudo_return_mult

    def update_exps(self):
        exps = super().update_exps()
        prs_reshaped = self.pseudo_returns_mult.transpose(1, 2).reshape(self.predictor.num_heads, -1, self.num_cumulants)
        exps.targets = prs_reshaped.transpose(0, 1)
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
        predictions = self.predictor(sb.obs)
        targets = sb.targets
        predictor_loss = []
        for h in range(self.predictor.num_heads):
            predictor_loss.append(self.predictor_criterion(predictions[h], targets[:, h, :]))
        intrinsic_loss = sum(predictor_loss)/len(predictor_loss)
        return intrinsic_loss