# LoRA-Finetuned Dialogue Summarizer Pipeline

This repository provides a complete pipeline for evaluating, fine-tuning, and testing a Mistral-based dialogue summarization model using LoRA adapters. The workflow is implemented in Python and is based on the HuggingFace and Unsloth ecosystems.

## Features
- **Baseline Evaluation:** Evaluate the performance of the base Mistral model on the DialogSum dataset.
- **Fine-tuning:** Fine-tune the model using LoRA adapters for improved summarization.
- **Evaluation:** Assess the performance of the fine-tuned model and compare it to the baseline.

## Requirements
- Python 3.8+
- CUDA-enabled GPU (recommended for training/inference)
- See `requirements.txt` for all dependencies

## Installation
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the full pipeline (baseline evaluation, fine-tuning, and evaluation):
```sh
python dialogue_summarizer_pipeline.py
```

## File Structure
- `dialogue_summarizer_pipeline.py`: Main pipeline script (class-based, modular).
- `requirements.txt`: List of required Python packages.

## Customization
You can modify parameters such as dataset size, model name, and training settings by editing the `DialogueSummarizerPipeline` class initialization in the script.

## Dataset
- Uses the [DialogSum dataset](https://huggingface.co/datasets/knkarthick/dialogsum) from HuggingFace Datasets.

## Model
- Base model: `unsloth/mistral-7b-instruct-v0.2-bnb-4bit`
- Fine-tuning uses LoRA adapters via Unsloth and TRL.

## Output
- Prints BERTScore F1 for both baseline and fine-tuned models.
- Saves LoRA adapters to disk after fine-tuning.

## Notes
- Ensure you have sufficient GPU memory for running large models.
- For best results, run on a machine with a CUDA-enabled GPU.

## License
MIT License

## Acknowledgements
- [Unsloth](https://github.com/unslothai/unsloth)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)
