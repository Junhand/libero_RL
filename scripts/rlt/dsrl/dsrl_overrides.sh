# Hydra overrides of the DSRL runs (training and evaluation share the model part).
DSRL_MODEL="actor.model.model_type=dsrl_mlp_policy actor.model.action_dim=32 actor.model.fixed_std=${DSRL_STD:-1.0} actor.model.dsrl_mean_max=${DSRL_MEAN_MAX:-1.5} rollout.rlt_feature_model.action_dim=7"
DSRL_TRAIN="$DSRL_MODEL algorithm.dsrl.enable=True algorithm.actor_weight_schedule.enable=False algorithm.q_weight=1.0 algorithm.bc_weight=${DSRL_REG:-0.05} algorithm.rlt_schedule.warmup_post_collect_updates=0"
DSRL_EVAL="rollout.model.model_type=dsrl_mlp_policy rollout.model.action_dim=32 rollout.model.fixed_std=${DSRL_STD:-1.0} rollout.model.dsrl_mean_max=${DSRL_MEAN_MAX:-1.5} rollout.rlt_feature_model.action_dim=7"
