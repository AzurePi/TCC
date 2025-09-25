import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer, \
    TrainerCallback, TrainerState, TrainerControl, AutoConfig


def compute_metrics(eval_pred):
    """
    Função usada pelo Trainer para calcular métricas extras em cada avaliação.
    Retorna um dict que será logado automaticamente (inclusive no CSVLoggerCallback).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_output_metrics(model_name, metrics, hparams=None):
    filename = "metrics.json"
    chave_modelo = f"{model_name} ({datetime.now()})"

    nova_entrada = {
        "hiperparametros": hparams if hparams else {},
        "metricas": metrics
    }

    if os.path.exists(filename):
        with open(filename, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    all_data[chave_modelo] = nova_entrada

    with open(filename, "w") as f:
        json.dump(all_data, f, indent=4)


class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_dir, model_name, **hparams):
        self.model_name = model_name
        self.output_dir = output_dir
        self.hparams = hparams
        self.start_time = None
        self.last_eval_metrics = {}  # Armazena as últimas métricas de avaliação

        # Os caminhos dos arquivos são definidos aqui, mas os arquivos são criados depois
        self.train_log_path = os.path.join(output_dir, "train_metrics.csv")
        self.eval_log_path = os.path.join(output_dir, "eval_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Chamado no início de cada execução de trainer.train()."""

        def _initialize_csv(path, headers):
            """Função auxiliar para criar um CSV com cabeçalho, sobrescrevendo se existir."""
            with open(path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        self.start_time = time.time()
        # Inicializa/sobrescreve os arquivos de log para garantir um registro limpo
        _initialize_csv(self.train_log_path, ["step", "metric", "value"])
        _initialize_csv(self.eval_log_path, ["step", "metric", "value"])
        print(f"Arquivos de log inicializados em '{self.output_dir}'. Logs antigos foram sobrescritos.")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Salva apenas métricas de treino (ex: loss, learning_rate)."""
        if logs is None or not state.is_world_process_zero:
            return

        step = state.global_step
        with open(self.train_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            for k, v in logs.items():
                if not k.startswith("eval_") and k != "epoch" and isinstance(v, (int, float)):
                    writer.writerow([step, k, v])

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None,
                    **kwargs):
        """Salva apenas métricas de avaliação."""
        if metrics is None or not state.is_world_process_zero:
            return

        # Armazena as métricas para uso no on_train_end
        self.last_eval_metrics = metrics.copy()

        step = state.global_step
        with open(self.eval_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                if k.startswith("eval_") and isinstance(v, (int, float)):
                    writer.writerow([step, k, v])

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Chamado no final do treino para salvar um resumo."""
        if not state.is_world_process_zero:
            return

        end_time = time.time()

        # Usa as métricas armazenadas em vez de ler o arquivo novamente
        final_metrics = self.last_eval_metrics
        final_metrics["tempo_total_segundos"] = round(end_time - self.start_time, 2)

        # Remove a métrica de época
        final_metrics.pop("epoch", None)

        save_output_metrics(
            model_name=self.model_name,
            metrics=final_metrics,
            hparams=self.hparams
        )


def get_model_and_processor(spec):
    model_id = spec["model"]
    is_finetuning = spec.get("finetuning", True)  # Assume finetuning se não especificado

    processor = AutoImageProcessor.from_pretrained(model_id)

    if is_finetuning:
        # Carrega o modelo com pesos pré-treinados
        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=model_id,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        print("Modelo carregado para fine-tuning.")
    else:
        # Carrega apenas a configuração do modelo e o inicializa com pesos aleatórios
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_id,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        model = AutoModelForImageClassification.from_config(config)
        print("Modelo inicializado com pesos aleatórios (treinamento do zero).")

    return model, processor


def get_trainer(model, dataset, spec, dataset_num):
    output_dir = Path(f"./results/{spec['name']}/dataset{dataset_num}")

    args = TrainingArguments(
        output_dir=output_dir.__str__(),
        eval_strategy="steps",
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        logging_dir="./logs",
        learning_rate=5e-5,
        overwrite_output_dir=True,
        seed=1234,

    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[
            CSVLoggerCallback(output_dir=output_dir / "metrics", model_name=spec["name"]),
        ],
        compute_metrics=compute_metrics
    )
