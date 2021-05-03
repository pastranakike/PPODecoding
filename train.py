from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from environment import DecodingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import torch
import os
from datetime import datetime
import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en").cuda()
train_data = pd.read_csv('/content/drive/MyDrive/Graduate/Spring2021/CS7650/final_project/training_data/training_data_tatoeba.tsv',
                         sep='\t', names=['tgt_tag', 'tgt', 'src_tag', 'src'])


env = DecodingEnv(model, tokenizer, train_data, 50, is_teacher_enforced=False)
check_env(env)

checkpoint_callback = CheckpointCallback(save_freq=2500, save_path='/content/drive/MyDrive/Graduate/Spring2021/CS7650/final_project/default_hparams',
                                         name_prefix='decoding_policy_ppo_default_hparam')

model = PPO("MlpPolicy", env, verbose=2, tensorboard_log='drive/MyDrive/policy_log', n_epochs=100, ent_coef=0.1, device="cuda")

model.learn(total_timesteps=100000000, callback=checkpoint_callback)
