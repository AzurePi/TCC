from pathlib import Path
from transformers import AutoModelForImageClassification, AutoImageProcessor
from carregar_datasets import carregar_datasets
from models import (
    get_model_and_processor,
    get_trainer_and_logger,
    get_latest_checkpoint,
    ModelWrapper
)
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter
import torch
import copy

dataset_nums = [1, 2, 3]
num_epochs = [1, 3, 5]
learning_rates = [1e-3, 1e-4, 1e-5]

model_specs = [
    {
        "name": "ConvNext - Transfer Learning",
        "model": "facebook/convnext-base-224-22k-1k",
        "type": "CNN",
        "transfer": True,
        "finetuning": False
    },
    {
        "name": "ConvNext - Ajuste Fino",
        "model": "facebook/convnext-base-224-22k-1k",
        "type": "CNN",
        "transfer": True,
        "finetuning": True
    },
    {
        "name": "ConvNext - Do Zero",
        "model": "facebook/convnext-base-224-22k-1k",
        "type": "CNN",
        "transfer": False,
        "finetuning": False
    },
    {
        "name": "Visual Transformer - Transfer Learning",
        "model": "google/vit-base-patch16-224",
        "type": "ViT",
        "transfer": True,
        "finetuning": False
    },
    {
        "name": "Visual Transformer - Ajuste Fino",
        "model": "google/vit-base-patch16-224",
        "type": "ViT",
        "transfer": True,
        "finetuning": True
    },
    {
        "name": "Visual Transformer - Do Zero",
        "model": "google/vit-base-patch16-224",
        "type": "ViT",
        "transfer": False,
        "finetuning": False
    }
]

if __name__ == "__main__":
    for spec in model_specs:
        print(f"{Fore.CYAN}\n------- Treinando modelo {spec['name']} ({spec['type']}) -------{Fore.RESET}")

        # Carrega modelo base uma única vez
        base_model, processor = get_model_and_processor(spec)

        # Registra o grafo apenas uma vez por modelo
        wrapped_model = ModelWrapper(base_model)
        writer = SummaryWriter(f"./runs/{spec['name'].replace(' ', '_').replace('-', '')}")
        writer.add_graph(wrapped_model, torch.randn(1, 3, 224, 224))
        writer.close()
        del wrapped_model

        for dataset_num in dataset_nums:
            dataset, class_names = carregar_datasets(
                dataset_num,
                0.3,
                processor,
                amostrar=False
            )

            label2id = {label: i for i, label in enumerate(class_names)}
            id2label = {i: label for i, label in enumerate(class_names)}

            print(f"{Fore.GREEN}{dataset}{Fore.RESET}")

            for learning_rate in learning_rates:
                for n_epochs in num_epochs:
                    print(f"\n--- Treinando por {n_epochs} epochs (lr={learning_rate}) ---")

                    output_dir = Path(f"./results/{spec['name']}/d{dataset_num}/lr_{learning_rate}_e_{n_epochs}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = get_latest_checkpoint(output_dir)

                    # Se existe checkpoint, retoma. Caso contrário, cria cópia do modelo base.
                    if checkpoint:
                        print(f"{Fore.YELLOW}Retomando checkpoint {checkpoint.name}{Fore.RESET}")
                        model = AutoModelForImageClassification.from_pretrained(checkpoint)
                    else:
                        model = copy.deepcopy(base_model)

                    model.config.label2id = label2id
                    model.config.id2label = id2label

                    trainer, csv_logger = get_trainer_and_logger(
                        model, dataset, dataset_num, spec, n_epochs, learning_rate, processor, output_dir
                    )

                    # Avaliação de baseline apenas em cenários de inicialização completamente aleatória
                    if not spec['transfer']:
                        csv_logger.set_baseline_mode(True)
                        trainer.evaluate(metric_key_prefix="baseline")
                        csv_logger.set_baseline_mode(False)

                    trainer.train()
                    trainer.evaluate()
