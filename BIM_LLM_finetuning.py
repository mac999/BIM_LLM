# title: LLM finetuning with PEFT
# description: BIM-based finetuning LLM (SLM) with PEFT with cheap price GPU(8GB) and TPU.
# author: taewook kang
# email: laputa99999@gmail.com
# date:
#   2024.6.1
#   2024.7.1
# license: MIT license
# usage:
# 1. pip install the below import packages
# 2. create account of huggingface website and input your huggingface API Key token in this source code. 
# 3. create account of wandb and input your wandb API key token in this source code. 
# 4. make QA dataset json file in dataset folder.
# 5. run this source code.
# reference:
#   https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
import os, pandas as pd, json, torch, wandb, logging, datetime, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from trl import SFTTrainer, setup_chat_format
from datasets import Dataset, load_dataset
from datasets import concatenate_datasets
from pathlib import Path
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	HfArgumentParser,
	TrainingArguments,
	pipeline,
)
from peft import (
	LoraConfig,
	PeftModel,
	prepare_model_for_kbit_training,
	get_peft_model,
)

# setup model file name
train_data_vagueness = False
base_model_file = 'Undi95/Meta-Llama-3-8B-hf' # Undi95/Meta-Llama-3-8B-hf
new_model_path = new_model_file = log_model_file = new_model_path_file = ''

# hyperparameter setting
epoch_num = 5
lora_r = 4
lora_alpha = 32

test_prompt_message = [{
		"role": "user",
		"content": 'What is the relationship between the level of detail in LOI and LOG?'
	}, 
	{
		"role": "user",
		"content": "What is the definition of information deliveries based on ISO 19650-1?"
	}
]

# create log file
login(token = '')   	# input your huggingface API Key token
log_file_path = 'finetuning.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
logger.info('Start finetuning LLM with PEFT')
logger.info(base_model_file)
logger.info(new_model_path_file)
logger.info(new_model_file)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
logger.addHandler(console_handler)

# utility function	
def set_model_file_path():
	global new_model_path_file, new_model_path, new_model_file, log_model_file, train_data_vagueness
	# setup new model file path
	current_time = datetime.datetime.now()
	formatted_time = current_time.strftime("%Y%m%d_%H%M")
	logger.info(formatted_time)

	new_model_path = './output_finetuning_model'
	if not os.path.exists(new_model_path):
		os.makedirs(new_model_path)
	log_model_file = f"llama-3-8B-hf-bim{formatted_time}"
	if train_data_vagueness == True:
		new_model_file = f"llama-3-8B-hf-bim{formatted_time}-v"
	else:
		new_model_file = f"llama-3-8B-hf-bim{formatted_time}" 
	new_model_path_file = f"{new_model_path}/{new_model_file}" # dataset_name = 'ruslanmv/ai-medical-chatbot'

custom_all_data = []
adj_words = ['and', 'or', 'to', 'in', 'on', 'at', 'of', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'shall', 'will', 'should', 'would', 'may', 'might', 'must', 'can', 'could', 'then']
def load_dataset_from_folder(folder_path):
	global custom_all_data, train_data_vagueness
	for file_name in os.listdir(folder_path):
		if file_name.endswith('.json'):
			file_path = os.path.join(folder_path, file_name)
			with open(file_path, 'r', encoding='utf-8') as file:
				data_list = json.load(file)
				for data in data_list['responses']:
					if train_data_vagueness == False:
						if int(data['vagueness']) == 1: 
							continue
						answer = data['answer']
						if isinstance(answer, str):
							answer = ' '.join([word for word in answer.split() if word not in adj_words])
							answer = answer.split()
						if len(answer) <= 2:
							continue

					# remove key index.
					if 'index' in data:
						del data['index']
					if 'vagueness' in data:
						del data['vagueness']
					if 'question' not in data:
						continue
					if 'answer' in data and isinstance(data['answer'], list):
						data['answer'] = ', '.join(data['answer'])
					# custom_all_data.extend(data)
					custom_all_data.append(data)

	df = pd.DataFrame(custom_all_data)
	df = df.astype(str)
	return Dataset.from_pandas(df)

def custom_format_chat_template(row, question_key='question', answer_key='answer'):
	global tokenizer
	row_json = [{"role": "user", "content": row[question_key]},
				{"role": "assistant", "content": row[answer_key]}]
	row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
	return row

def generate_answer(message_list, model, tokenizer):
	for message in message_list:
		messages = [message]

		logger.info(messages)

		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
		outputs = model.generate(**inputs, max_length=256, num_return_sequences=1, temperature=0.1) # max_length=150, 
		text = tokenizer.decode(outputs[0], skip_special_tokens=True)
		logger.info(text.split("assistant")[1])

# train and test function
def train_model():
	global base_model_file, new_model_path_file, new_model_path, new_model_file, log_model_file, tokenizer
	global test_prompt_message, test_prompt_message
	global epoch_num, lora_r, lora_alpha

	logger.info(f'epoch_num: {epoch_num}')
	logger.info(f'lora_r: {lora_r}')
	logger.info(f'lora_alpha: {lora_alpha}')

	wandb.login(key='')		# input your wandb API key token
	run = wandb.init(
		project=log_model_file,
		job_type="train_modeling",
		anonymous="allow"
	) 

	# QLoRA setting
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True, # 4bit quantumnization
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_use_double_quant=True,
	)

	base_model = AutoModelForCausalLM.from_pretrained(
		base_model_file,
		quantization_config=bnb_config,
		device_map="auto",
		attn_implementation="eager", 
		# local_files_only=False
	)

	# load tokenizer and base model 
	tokenizer = AutoTokenizer.from_pretrained(base_model_file) # , local_files_only=False)
	base_model, tokenizer = setup_chat_format(base_model, tokenizer)

	# before fine tuning test.
	logger.info('train. before fine tuning')
	generate_answer(test_prompt_message, base_model, tokenizer)

	# Lora PEFT setting
	peft_config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		lora_dropout=0.05,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
	)
	peft_model = get_peft_model(base_model, peft_config) 

	# dataset = load_dataset(dataset_name, split="all")
	global custom_all_data
	custom_dataset = load_dataset_from_folder('./dataset') # user custom dataset usage
	custom_dataset = custom_dataset.map(
		custom_format_chat_template,
		num_proc=1,
	)
	if len(custom_dataset['text']) > 100: # test
		custom_question = custom_dataset['text'][100]
		print(custom_question)
	dataset = custom_dataset # concatenate_datasets([dataset, custom_dataset])
	# dataset = dataset.shuffle(seed=65).select(range(1000))  # Use only 1000 examples for fine-tuning
	print(dataset['text'][3])

	dataset = dataset.train_test_split(test_size=0.1) 

	train_modeling_arguments = TrainingArguments(
		output_dir=new_model_path_file,
		per_device_train_batch_size=1,
		per_device_eval_batch_size=1,
		gradient_accumulation_steps=2,
		optim="paged_adamw_32bit",
		num_train_epochs=3, # epoch_num, # 15, 
		evaluation_strategy="steps",
		eval_steps=0.2,
		logging_steps=1,
		warmup_steps=10,
		logging_strategy="steps",
		learning_rate=2e-4,
		fp16=False,
		bf16=False,
		group_by_length=True,
		report_to="wandb"
	)

	# superviser fine tuning settting
	train_modeler = SFTTrainer(
		model=peft_model,
		train_dataset=dataset["train"],
		eval_dataset=dataset["test"],
		peft_config=peft_config,
		max_seq_length=512,
		dataset_text_field="text",
		tokenizer=tokenizer,
		args=train_modeling_arguments,
		packing= False,
	) 

	# train
	train_modeler.train()

	# save train model model
	train_modeler.model.save_pretrained(new_model_path_file)

	# change working directory
	original_path = os.getcwd()
	os.chdir(new_model_path)
	peft_model = PeftModel.from_pretrained(base_model, new_model_file) # , device_map={"":0})
	# peft_model = peft_model.merge_and_unload() # if 4bit then Cannot merge LORA layers when the model is loaded in 8-bit mode 
	# peft_model.save_pretrain_modeled(new_model_path_file)
	peft_model.push_to_hub(new_model_file, use_temp_dir=False)
	os.chdir(original_path)
	
	# end W&B log 
	run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
	summary = run.summary

	logger.info("Performance Summary from W&B:")
	logger.info(f'wandb project: {wandb.run.project}')
	logger.info(f'wandb run id: {wandb.run.id}')
	logger.info(f'wandb run name: {wandb.run.name}')
	for key, value in summary.items():
		logger.info(f"{key}: {value}")	

	wandb.finish()
	peft_model.config.use_cache = True

	# test fine tuning model
	logger.info('train. after fine tuning')
	peft_model, tokenizer = setup_chat_format(peft_model, tokenizer)
	peft_model.eval()
	generate_answer(test_prompt_message, peft_model, tokenizer)

	base_model.cpu()
	peft_model.cpu()
	del base_model, peft_model, tokenizer
	gc.collect() 	
	torch.cuda.empty_cache() 

def test_original_llama():
	global base_model_file
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True, # 4bit 양자화
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_use_double_quant=True,
	)
	model = AutoModelForCausalLM.from_pretrained(
		base_model_file,
		quantization_config=bnb_config,
		device_map="auto",
		attn_implementation="eager",
		local_files_only=False
	)
	tokenizer = AutoTokenizer.from_pretrained(base_model_file, local_files_only=False)
	model, tokenizer = setup_chat_format(model, tokenizer)
	model.eval()

	logger.info('test. before fine tuning')
	generate_answer(test_prompt_message, model, tokenizer)

	model.cpu()
	del model, tokenizer
	gc.collect() 
	torch.cuda.empty_cache()

def test_peft_llama():
	global new_model_path_file, base_model_file
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True, # 4bit 양자화
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_use_double_quant=True,
	)
	base_model = AutoModelForCausalLM.from_pretrained(
		base_model_file,
		quantization_config=bnb_config,
		device_map="auto",
		attn_implementation="eager",
		local_files_only=False
	)
	tokenizer = AutoTokenizer.from_pretrained( "Undi95/Meta-Llama-3-8B-hf", local_files_only=False)
	original_path = os.getcwd()
	os.chdir(new_model_path)
	peft_model = PeftModel.from_pretrained(base_model, new_model_file)
	os.chdir(original_path)
	peft_model, tokenizer = setup_chat_format(peft_model, tokenizer)
	peft_model.eval()	

	logger.info('perf. after fine tuning')
	generate_answer(test_prompt_message, peft_model, tokenizer)

	base_model.cpu()
	peft_model.cpu()
	del base_model, peft_model, tokenizer
	gc.collect() 	
	torch.cuda.empty_cache()

if __name__ == "__main__":
	for epoch_num in [5]: # [5, 10, 15]: # for parameter tuning
		for lora_r in [16]:
			train_data_vagueness = True 
			set_model_file_path()

			train_model()

			# test the model between original model and fine tuning model
			test_original_llama()
			test_peft_llama()
