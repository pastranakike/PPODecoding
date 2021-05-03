import torch
import numpy as np
import gym
from gym import spaces
import datasets
import random
import math
from sentence_transformers import SentenceTransformer, util

class DecodingEnv(gym.Env):
    def __init__(self, seq2seq, tokenizer, dataset, max_len, max_eps=0.95, min_eps=0.05, eps_decay=50000, is_teacher_enforced=False):
        """"
          OpenAi gym-like interface for sequence-to-sequence decoding. At a given timestep, the agent produces an action, which is repre
          sented as a token in the sequence, compute the reward associated with the current state of the sentece and query the seq2seq model
          for the next decoding hidden state.
          @param seq2seq: MarianMTModel (or any other seq2seq model from HuggingFace)
          @param tokenizer: MarianTokenizer (or any other HuggingFace tokenizer)
          @param dataset: pandas.DataFrame
          @param max_len: int
          @param max_eps: int
          @param min_eps: int
          @param eps_decay: int
          @param is_teacher_enforced: bool
        """"
        super(DecodingEnv, self).__init__()
        #Stuff for openAI gym
        self.action_space = gym.spaces.Discrete(tokenizer.vocab_size)
        self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'),
                                        shape=(1, 512), dtype=np.double)

        #Init the models
        self.seq2seq = seq2seq
        self.tokenizer = tokenizer
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        ##This for handling the dataset
        src_text = list(dataset['src'].values)
        self.src = tokenizer(src_text, return_tensors="pt", padding=True)
        tgt_text = list(dataset['tgt'].values)
        self.tgt = tgt_text
        self.tgt_tokenized = tokenizer(tgt_text, return_tensors="pt", padding=True)
        self.cos_sim_model = SentenceTransformer('paraphrase-distilroberta-base-v1').to(self.device)

        #Set environment metadata
        self.max_len = max_len
        self.curr_trial = -1
        self.is_teacher_enforced = is_teacher_enforced
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay

    def step(self, action):
        """
          Produces a step in the environment by appending a new token into the current sentence. If is_teacher_enforced=True, it will use the ground
          truth to set the next state. This is useful to avoid compounding errors at the beggining of the training. There is a epsilon-decay associated
          with how often with query the ground-truth.
        """
        token = torch.tensor([[action]]).to(self.device)
        self.curr_stc = torch.cat([self.curr_stc, token], dim=1)
        if token.item() == self.tokenizer.eos_token_id or self.max_len == self.step_counter:
            reward = self._calculate_terminal_reward() + self._calculate_slack_reward()
            done = True
        else:
            reward = self._calculate_slack_reward()
            done = False
        
        p = random.random()
        eps_decayed = self.min_eps + (self.max_eps - self.min_eps) * math.exp(-1 * self.curr_trial / self.eps_decay)
        if self.is_teacher_enforced and p <= eps_decayed:
            decoder_output = self.seq2seq(input_ids=self.curr_src, decoder_input_ids=self.curr_tgt_tokenized[:, 0:self.step_counter+1], output_hidden_states=True)
        else:
            decoder_output = self.seq2seq(input_ids=self.curr_src, decoder_input_ids=self.curr_stc, output_hidden_states=True)
        self.curr_state = decoder_output.decoder_hidden_states[-1][:, -1, :]
        self.step_counter += 1
        reward = reward.item()
        return self.curr_state.cpu().detach().numpy(), reward, done, {}
    
    def reset(self):
        """
          Reset the environment by emptying the current sentence and query the next sentence in the dataset.
        """
        self.step_counter = 0
        self.curr_token = self.tokenizer.bos_token_id
        self.curr_trial += 1
        inx = random.randint(0, len(self.src) - 1)
        self.curr_tgt = self.tgt[inx]
        self.curr_tgt_tokenized = self.tgt_tokenized['input_ids'][inx].to(self.device).unsqueeze(0)
        self.curr_src = self.src['input_ids'][inx].to(self.device).unsqueeze(0)
        self.curr_stc = self.tokenizer('<pad>', add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
        decoder_output = self.seq2seq(input_ids=self.curr_src, decoder_input_ids=self.curr_stc.unsqueeze(0), output_hidden_states=True)
        self.curr_state = decoder_output.decoder_hidden_states[-1][:, -1, :]
        self.bleu_score = datasets.load_metric('sacrebleu')
        return self.curr_state.cpu().detach().numpy()

    def _calculate_slack_reward(self):
        """
          Returns the cosine similarity between two sentence using DistilRoBERTa embeddings.
        """
        curr_stc = ' '.join([self.tokenizer.decode(t, skip_special_tokens=True) for t in self.curr_stc])
        tgt_to_step = self.curr_tgt.split()
        sentences2 = ' '.join(tgt_to_step[0:self.step_counter+1])
        embeddings1 = self.cos_sim_model.encode(curr_stc, convert_to_tensor=True).cuda()
        embeddings2 = self.cos_sim_model.encode(sentences2, convert_to_tensor=True).cuda()

        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        return -(1 - cosine_scores)

    def _calculate_terminal_reward(self):
        """
          Returns the BLEU score between the predicted sentence and ground-truth once the EOS token is generated.
        """
        prediction = [self.tokenizer.decode(t, skip_special_tokens=True) for t in self.curr_stc]
        target = self.curr_tgt
        target = [target.split()]
        self.bleu_score.add_batch(predictions=prediction, references=target)
        return self.bleu_score.compute()['score']
