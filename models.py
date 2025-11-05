import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import (
    AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer,
    TrainerCallback, TrainerState, TrainerControl, AutoConfig, default_data_collator, PreTrainedModel,
    BaseImageProcessor, PrinterCallback, ProgressCallback
)


def compute_metrics(eval_pred):
    """
    Função usada pelo Trainer para calcular métricas extras em cada avaliação.
    Retorna um dict que será logado automaticamente (inclusive no CSVLoggerCallback).
    """
    logits, labels = eval_pred
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    preds = logits.argmax(axis=-1)

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
        self.output_dir = output_dir
        self.model_name = model_name
        self.hparams = hparams

        self.start_time = None
        self.last_eval_metrics = {}
        self.is_baseline = False

        self.train_log_path = os.path.join(output_dir, "train_metrics.csv")
        self.eval_log_path = os.path.join(output_dir, "eval_metrics.csv")
        self.baseline_log_path = os.path.join(output_dir, "baseline_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)

    def set_baseline_mode(self, mode: bool):
        """Permite marcar que a próxima avaliação é baseline."""
        self.is_baseline = mode

    def _initialize_csv(self, path, headers):
        """Função auxiliar para criar um CSV com cabeçalho, sobrescrevendo se existir."""
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Chamado no início de cada execução de trainer.train()."""
        self.start_time = time.time()
        # Inicializa/sobrescreve os arquivos de log para garantir um registro limpo
        if not os.path.exists(self.train_log_path):
            self._initialize_csv(self.train_log_path, ["step", "metric", "value"])
            print(
                f"{Fore.BLUE}Arquivo de log inicializado em '{self.train_log_path}'. Logs antigos foram sobrescritos.{Fore.RESET}")
        if not os.path.exists(self.eval_log_path):
            self._initialize_csv(self.eval_log_path, ["step", "metric", "value"])
            print(
                f"{Fore.BLUE}Arquivo de log inicializado em '{self.eval_log_path}'. Logs antigos foram sobrescritos.{Fore.RESET}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Salva apenas métricas de treino."""
        if logs is None or not state.is_world_process_zero:
            return

        step = state.global_step
        with open(self.train_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            for k, v in logs.items():
                if not k.startswith("eval_") and not k.startswith("baseline_") and k != "epoch" and isinstance(v, (int,
                                                                                                                   float)):
                    writer.writerow([step, k, v])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or not state.is_world_process_zero:
            return

        if self.is_baseline:
            self._initialize_csv(self.baseline_log_path, ["step", "metric", "value"])
            target_path = self.baseline_log_path
            prefix = "baseline_"
        else:
            self.last_eval_metrics = metrics.copy()
            target_path = self.eval_log_path
            prefix = "eval_"

        step = state.global_step

        with open(target_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                if k.startswith(prefix) and isinstance(v, (int, float)):
                    writer.writerow([step, k, v])

        if self.is_baseline:
            print(f"{Fore.BLUE}Baseline salvo em {self.baseline_log_path}{Fore.RESET}")
            self.set_baseline_mode(mode=False)  # reseta para não poluir avaliações seguintes

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Chamado no final do treino para salvar um resumo."""
        if not state.is_world_process_zero:
            return

        end_time = time.time()

        # Usa as métricas armazenadas em vez de ler o arquivo novamente
        final_metrics = self.last_eval_metrics
        final_metrics["tempo_total_segundos"] = round(end_time - self.start_time, 2)

        # Remove a métrica de época, se houver
        final_metrics.pop("epoch", None)

        save_output_metrics(
            model_name=self.model_name,
            metrics=final_metrics,
            hparams=self.hparams
        )


class ProgressOverrideCallback(ProgressCallback):
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)


from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
from colorama import Fore


def get_model_and_processor(spec: dict):
    model_id = str(spec["model"])
    processor = AutoImageProcessor.from_pretrained(model_id)

    transfer = spec.get("transfer", True)
    finetuning = spec.get("finetuning", True)

    if not transfer:
        # Treinamento do zero
        config = AutoConfig.from_pretrained(model_id, num_labels=2)
        model = AutoModelForImageClassification.from_config(config)
        print(f"{Fore.LIGHTGREEN_EX}Modelo inicializado do zero.{Fore.RESET}")
        return model, processor

    # Se transfer=True, carregamos modelo pré-treinado
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    if finetuning:
        # Fine-tuning total
        print(f"{Fore.LIGHTGREEN_EX}Modelo carregado para fine-tuning completo.{Fore.RESET}")
        return model, processor

    # Caso transfer=True e finetuning=False → Transfer Learning (congelar backbone)
    print(f"{Fore.LIGHTGREEN_EX}Modelo carregado para transfer learning (backbone congelado).{Fore.RESET}")

    # Congela todos os parâmetros
    for param in model.parameters():
        param.requires_grad = False

    # Apenas a cabeça final fica treinável
    # A cabeça usada pelos modelos da HF para classificação é sempre model.classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{Fore.LIGHTGREEN_EX}Treinável: {name}{Fore.RESET}")

    return model, processor


def get_trainer_and_logger(
        model,
        dataset,
        dataset_num,
        spec: dict,
        num_epochs: int,
        learning_rate: float,
        processor,
        output_dir
) -> (Trainer, CSVLoggerCallback):
    args = TrainingArguments(
        output_dir=str(output_dir),
        save_strategy="epoch",  # salva ao final de cada epoch
        save_total_limit=3,  # mantém últimos checkpoints
        load_best_model_at_end=False,
        overwrite_output_dir=False,  # não sobrescreve checkpoints anteriores
        eval_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        dataloader_num_workers=4,
        seed=1234,
    )

    csv_logger = CSVLoggerCallback(output_dir=output_dir, model_name=spec["name"])

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
        data_collator=default_data_collator,
        callbacks=[csv_logger],
        compute_metrics=compute_metrics
    )

    progress_callback = (next(filter(lambda x: isinstance(x, ProgressCallback), trainer.callback_handler.callbacks),
                              None))
    trainer.remove_callback(progress_callback)
    trainer.add_callback(ProgressOverrideCallback)

    trainer.remove_callback(PrinterCallback)

    return trainer, csv_logger


def get_latest_checkpoint(output_dir: Path):
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
    return checkpoints[-1] if checkpoints else None


class ModelWrapper(nn.Module):
    """
    Wrapper para pegar só o tensor dentre os outputs do modelo huggingface; para criar diagrama da arquitetura.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out['logits']  # só o tensor que interessa
