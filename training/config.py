from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_seq_length: int = 512

    # QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    fp16: bool = True

    # Paths
    output_dir: str = "./outputs/finetuned"
    data_train: str = "./data/train.jsonl"
    data_val: str = "./data/val.jsonl"

    # Logging
    logging_steps: int = 20
    eval_steps: int = 100
    save_strategy: str = "epoch"
    report_to: str = "none"   # set to "wandb" to enable W&B logging
