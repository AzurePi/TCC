import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator


def useless_metric(metric: str):
    if metric == "total_flos":
        return True
    if metric == "train_loss":
        return True
    if metric.startswith('epoch'):
        return True
    if metric == 'grad_norm':
        return True
    if metric.endswith('runtime'):
        return True
    if metric.endswith('per_second'):
        return True
    return False


def carregar_logs_csv(train_log_path: str, eval_log_path: str):
    """Carrega métricas de treino e validação a partir dos CSVs do callback."""
    treino = {}
    validacao = {}

    # Treino
    if os.path.exists(train_log_path):
        with open(train_log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["step"])
                metric = row["metric"]
                value = float(row["value"])
                if not useless_metric(metric):
                    treino.setdefault(metric, []).append((step, value))

    # Validação
    if os.path.exists(eval_log_path):
        with open(eval_log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["step"])
                metric = row["metric"]
                value = float(row["value"])
                if not useless_metric(metric):
                    clean_metric = metric.removeprefix("eval_")
                    validacao.setdefault(clean_metric, []).append((step, value))

    # Converter para arrays alinhados
    treino_alinhado = {m: [v for _, v in sorted(vals)] for m, vals in treino.items()}
    validacao_alinhado = {m: [v for _, v in sorted(vals)] for m, vals in validacao.items()}

    return treino_alinhado, validacao_alinhado


def grafico_aux(titulo: str, treino: list[float] | None, validacao: list[float] | None,
                ylabel: str, output_path: Path) -> None:
    """Plota uma curva, podendo ser treino, validação, ou ambos."""
    plt.title(titulo)

    max_len = 0
    max_val = 0.0

    if treino:
        plt.plot(treino, label="Treino")
        max_len = max(max_len, len(treino))
        max_val = max(max_val, max(treino))

    if validacao:
        plt.plot(validacao, label="Validação")
        max_len = max(max_len, len(validacao))
        max_val = max(max_val, max(validacao))

    plt.xlabel("Step")
    plt.xticks(np.arange(0, max_len, step=max(1, max_len // 5)))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(ylabel)

    if ylabel.lower().startswith("perda") or ylabel.lower().startswith("loss"):
        if max_val >= 4.5:
            plt.ylim(0, 5)
            plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
        else:
            plt.ylim(0, max_val + 0.1 * abs(max_val))
            plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
    else:
        plt.ylim(0, 1.0)
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend(shadow=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plotar_graficos_from_logs(output_dir: str, model_name: str):
    """
    Lê os logs CSV do callback e gera gráficos de perda e acurácia.

    :param output_dir: diretório onde estão 'train_metrics.csv' e 'eval_metrics.csv'
    :param model_name: usado para nomear os arquivos de saída
    """
    train_log = os.path.join(output_dir, "metrics/train_metrics.csv")
    eval_log = os.path.join(output_dir, "metrics/eval_metrics.csv")

    treino, validacao = carregar_logs_csv(train_log, eval_log)

    graphs_dir = Path(output_dir) / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    # Identifica todas as métricas únicas
    todas_metricas = set(treino.keys()) | set(validacao.keys())

    for metric in sorted(list(todas_metricas)):
        train_data = treino.get(metric, None)
        # Procura a métrica de validação pelo nome com ou sem o prefixo
        eval_data = validacao.get(metric, None) or validacao.get("eval_" + metric, None)

        if train_data or eval_data:
            titulo = f"{metric.replace('_', ' ').capitalize()} ({model_name})"
            ylabel = metric.replace('_', ' ').capitalize()

            # Evita o prefixo "eval_" na label do eixo Y
            if ylabel.lower().startswith("eval"):
                ylabel = ylabel.removeprefix("eval ").strip()

            grafico_aux(
                titulo=titulo,
                treino=train_data,
                validacao=eval_data,
                ylabel=ylabel,
                output_path=graphs_dir / f"{model_name}_{metric}.svg"
            )


def carregar_resultados_json(filename="metrics.json"):
    """Lê o arquivo de métricas finais salvo pelo callback."""
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        return json.load(f)


def extrair_runtime(log_path: str, runtime_metric: str) -> float | None:
    """
    Extrai o valor de uma métrica de runtime de um arquivo CSV de logs.
    Presume que a métrica de runtime aparece uma única vez.
    """
    if not os.path.exists(log_path):
        return None

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric")
            if metric == runtime_metric:
                try:
                    return float(row["value"])
                except (ValueError, TypeError):
                    return None

    return None


if __name__ == "__main__":
    from treinamento import model_specs, dataset_nums

    for spec in model_specs:
        for dataset_num in dataset_nums:
            plotar_graficos_from_logs(
                output_dir=f"./results/{spec['name']}/dataset{dataset_num}",
                model_name=spec["name"]
            )
