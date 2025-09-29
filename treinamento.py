from carregar_datasets import carregar_datasets
from models import get_model_and_processor, get_trainer
from transformers.utils import logging

logging.set_verbosity_error()

model_specs = [
    {
        "name": "ConvNext - Transfer Learning",
        "model": "facebook/convnext-base-224-22k-1k",
        "type": "CNN",
        "finetuning": True
    },
    {
        "name": "ConvNext - Do Zero",
        "model": "facebook/convnext-base-224-22k-1k",
        "type": "CNN",
        "finetuning": False
    },
    {
        "name": "Visual Transformer - Transfer Learning",
        "model": "google/vit-base-patch16-224",
        "type": "ViT",
        "finetuning": True
    },
    {
        "name": "Visual Transformer - Do Zero",
        "model": "google/vit-base-patch16-224",
        "type": "ViT",
        "finetuning": False
    }
]

dataset_nums = [1, 2]

if __name__ == '__main__':
    for spec in model_specs:
        print(f"\n{'-' * 7} Treinando modelo {spec['name']} ({spec['type']}) {'-' * 7}")

        for dataset_num in dataset_nums:
            model, processor = get_model_and_processor(spec)

            dataset, class_names = carregar_datasets(dataset_num, 0.3, processor)

            model.config.label2id = {label: i for i, label in enumerate(class_names)}
            model.config.id2label = {i: label for i, label in enumerate(class_names)}

            print(dataset)

            trainer = get_trainer(model, dataset, spec, dataset_num, processor)

            trainer.train()

            trainer.evaluate()

            print()
