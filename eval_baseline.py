from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import tqdm
import datasets

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en").cuda()

tatoeba = pd.read_csv('/content/drive/MyDrive/Graduate/Spring2021/CS7650/final_project/test_data/test.txt', sep='\t', names=['tgt_tag', 'src_tag', 'tgt', 'src'])
en = pd.read_csv('/content/drive/MyDrive/Graduate/Spring2021/CS7650/final_project/test_data/nc-devtest2007.en', sep='NONE', names=['tgt'])
es = pd.read_csv('/content/drive/MyDrive/Graduate/Spring2021/CS7650/final_project/test_data/nc-devtest2007.es', sep='NONE', names=['src'])
wmt07 = pd.concat([en, es], axis=1)
test_set = tatoeba

src_text = list(test_set['src'].values)
ref_text = list(test_set['tgt'].values)
ref_text = [[stc] for stc in ref_text]
input_src = tokenizer(src_text, return_tensors="pt", padding=True)
num_stc = len(src_text)
batch_size = 100

pred_stc_greedy = []
pred_stc_k_beams = []
for i in tqdm.notebook.tqdm(range(0, num_stc, batch_size), leave=False):
    if i + batch_size >= num_stc:
        input_batch = input_src['input_ids'][i:].cuda()
        attn_batch = input_src['attention_mask'][i:].cuda()
    else:
        input_batch = input_src['input_ids'][i:i+batch_size].cuda()
        attn_batch = input_src['attention_mask'][i:i+batch_size].cuda()
    pred = model.generate(input_ids=input_batch, attention_mask=attn_batch, num_beams=1, num_beam_groups=1, do_sample=False)
    pred_stc_greedy += [tokenizer.decode(t, skip_special_tokens=True) for t in pred]

batch_size = int(batch_size / 2)
for i in tqdm.notebook.tqdm(range(0, num_stc, batch_size), leave=False):
    if i + batch_size >= num_stc:
        input_batch = input_src['input_ids'][i:].cuda()
        attn_batch = input_src['attention_mask'][i:].cuda()
    else:
        input_batch = input_src['input_ids'][i:i+batch_size].cuda()
        attn_batch = input_src['attention_mask'][i:i+batch_size].cuda()
    pred = model.generate(input_ids=input_batch, attention_mask=attn_batch, num_beams=10, num_beam_groups=1, do_sample=False)
    pred_stc_k_beams += [tokenizer.decode(t, skip_special_tokens=True) for t in pred]
    
test_set['greedy_out'] = pred_stc_greedy
test_set['beam_search_out'] = pred_stc_k_beams

metric_greedy = datasets.load_metric('sacrebleu')
metric_greedy.add_batch(predictions=list(test_set['greedy_out'].values), references=ref_text)
greedy_results = metric_greedy.compute()
print("GREEDY")
print(greedy_results)
metric_beam = datasets.load_metric('sacrebleu')
metric_beam.add_batch(predictions=list(test_set['beam_search_out'].values), references=ref_text)
beam_results = metric_beam.compute()
print("BEAM")
print(beam_results)
