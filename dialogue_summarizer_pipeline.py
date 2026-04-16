# dialogue_summarizer_pipeline.py
"""
Combined script for baseline evaluation, fine-tuning, and evaluation of a Mistral-based dialogue summarizer.
Refactored into a class-based structure.
"""

import torch
import evaluate
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback

class DialogueSummarizerPipeline:
    def __init__(self, test_data_size=30, train_size=300, validation_size=50, lora_model_name="dialogue-summarizer-mistral"):
        self.test_data_size = test_data_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.lora_model_name = lora_model_name
        self.dataset = None
        self.test_dataset = None
        self.test_dialogues = None
        self.test_summaries = None
        self.model = None
        self.tokenizer = None
        self.EOS_TOKEN = None
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"""

    def load_data(self):
        print("Loading datasets...")
        self.dataset = load_dataset("knkarthick/dialogsum")
        self.test_dataset = self.dataset['test'].shuffle(seed=42).select(range(self.test_data_size))
        self.test_dialogues = [sample['dialogue'] for sample in self.test_dataset]
        self.test_summaries = [sample['summary'] for sample in self.test_dataset]

    def load_base_model(self, max_seq_length=4096):
        print("Loading base Mistral model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(self.model)
        self.EOS_TOKEN = self.tokenizer.eos_token

    def baseline_evaluation(self):
        print("\n=== Baseline Summarizer Evaluation ===\n")
        summarization_prompt_template = """[INST]\nSummarize the dialogue mentioned in the user input. Be specific and concise in your summary.\nEnsure that you retain the entities mentioned in the dialogue in your summary.\n\n### User Input:\n{dialogue}\n[/INST]\n"""
        predicted_summaries = []
        print("Running baseline inference...")
        for gold_dialogue in tqdm(self.test_dialogues):
            try:
                prompt = summarization_prompt_template.format(dialogue=gold_dialogue)
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    temperature=0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                prediction = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1]:],
                    skip_special_tokens=True,
                    cleanup_tokenization_spaces=True
                )
                predicted_summaries.append(prediction)
            except Exception as e:
                print(e)
                continue
        print("Evaluating baseline model...")
        bert_scorer = evaluate.load("bertscore")
        score = bert_scorer.compute(
            predictions=predicted_summaries,
            references=self.test_summaries,
            lang='en',
            rescale_with_baseline=True
        )
        baseline_f1 = sum(score['f1'])/len(score['f1'])
        print(f"Baseline BERTScore F1: {baseline_f1:.4f}\n")
        return baseline_f1

    def finetune(self):
        print("\n=== Fine-tuning Mistral Summarizer ===\n")
        train_dataset = self.dataset['train'].shuffle(seed=42).select(range(self.train_size))
        validation_dataset = self.dataset['validation'].shuffle(seed=42).select(range(self.validation_size))
        def prompt_formatter(example, prompt_template, eos_token):
            instruction = 'Write a concise summary of the following dialogue.'
            dialogue = example["dialogue"]
            summary = example["summary"]
            formatted_prompt = prompt_template.format(instruction, dialogue, summary) + eos_token
            return {'formatted_prompt': formatted_prompt}
        formatted_training_dataset = train_dataset.map(
            lambda ex: prompt_formatter(ex, self.alpaca_prompt, self.EOS_TOKEN)
        )
        formatted_validation_dataset = validation_dataset.map(
            lambda ex: prompt_formatter(ex, self.alpaca_prompt, self.EOS_TOKEN)
        )
        print("Attaching LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing=True,
            random_state=42,
            loftq_config=None
        )
        print("Starting fine-tuning...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_training_dataset,
            eval_dataset=formatted_validation_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            dataset_text_field = "formatted_prompt",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            args = TrainingArguments(
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=10,
                evaluation_strategy="epoch",
                save_strategy='epoch',
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                greater_is_better=False,
                learning_rate=5e-5,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="paged_adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
                output_dir="outputs"
            )
        )
        trainer.train()
        print("Fine-tuning complete.\n")
        self.model.save_pretrained(self.lora_model_name)
        print(f"LoRA adapters saved to {self.lora_model_name}/\n")

    def load_finetuned_model(self, max_seq_length=2048):
        print("Loading fine-tuned model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.lora_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(self.model)

    def evaluate_finetuned(self):
        print("\n=== Evaluation of Fine-tuned Summarizer ===\n")
        predicted_summaries_ft = []
        instruction = 'Write a concise summary of the following dialogue.'
        for gold_dialogue in tqdm(self.test_dialogues):
            try:
                prompt = self.alpaca_prompt.format(
                    instruction,
                    gold_dialogue,
                    ""
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                prediction = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1]:],
                    skip_special_tokens=True,
                    cleanup_tokenization_spaces=True
                )
                predicted_summaries_ft.append(prediction)
            except Exception as e:
                print(e)
                continue
        print("Evaluating fine-tuned model...")
        bert_scorer = evaluate.load("bertscore")
        score_ft = bert_scorer.compute(
            predictions=predicted_summaries_ft,
            references=self.test_summaries,
            lang='en',
            rescale_with_baseline=True
        )
        finetuned_f1 = sum(score_ft['f1'])/len(score_ft['f1'])
        print(f"Fine-tuned BERTScore F1: {finetuned_f1:.4f}\n")
        return finetuned_f1

def main():
    pipeline = DialogueSummarizerPipeline()
    pipeline.load_data()
    pipeline.load_base_model()
    pipeline.baseline_evaluation()
    pipeline.finetune()
    pipeline.load_finetuned_model()
    pipeline.evaluate_finetuned()
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
