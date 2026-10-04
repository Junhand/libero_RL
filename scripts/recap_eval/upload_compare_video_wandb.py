import wandb
run = wandb.init(project="libero_RL", entity="junhand", name="recap-noprompt-vs-base-init14",
    job_type="demo-video",
    notes=("LIBERO-10 Task 0、初期状態(trial id) 14。同じ初期状態で、ベース(smolvla_libero)は失敗、RECAP(global_step_2000、プロンプトなしで評価)は成功した例。"
           "左側の環境(偶数の trial id)を切り出して 512x512 に拡大。注意: これは、ベースが失敗して RECAP が成功した 11 個の初期状態から選んだ 1 例で、"
           "全体では、ベース 18/50(0.36、2 環境、noise_seed=0)、RECAP 25/50(0.50、2 環境、noise_seed=null)で、差は有意ではない(Fisher p≈0.2)。"
           "逆に、ベースが成功して RECAP が失敗した初期状態も 4 個ある(4, 23, 31, 40)。"),
    config={"task": "LIBERO-10 Task 0", "trial_id": 14,
            "base_model": "models/smolvla_libero", "base_eval": "2 envs x 25 passes, noise_seed=0, advantage_prompt=false",
            "recap_ckpt": "RLinf/logs/20261003-20:10:02-libero10_task0_recap_smolvla/step4_recap_noweight_resume1000/checkpoints/global_step_2000",
            "recap_eval": "2 envs x 25 passes, noise_seed=null, advantage_prompt=false",
            "base_success_once": "18/50 (0.36)", "recap_success_once": "25/50 (0.50)",
            "base_fail_recap_success_trials": [0, 7, 8, 14, 15, 24, 34, 37, 38, 47, 48],
            "base_success_recap_fail_trials": [4, 23, 31, 40]})
wandb.log({"video/base_smolvla_FAIL": wandb.Video("init14_base_smolvla.mp4", fps=10, format="mp4"),
           "video/recap_step2000_noprompt_SUCCESS": wandb.Video("init14_recap_step2000_noprompt.mp4", fps=10, format="mp4")})
print(run.url)
run.finish()
