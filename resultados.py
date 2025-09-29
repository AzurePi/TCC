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


def extrair_metrica_final(log_path: str, metric_name: str) -> float | None:
    """
    Extrai o valor de uma métrica final específica de um arquivo CSV de logs.

    :param log_path: Caminho completo para o arquivo CSV (geralmente eval_metrics.csv).
    :param metric_name: O nome exato da métrica a ser extraída (e.g., 'eval_accuracy').
    :return: O valor da métrica como float ou None se não encontrado.
    """
    if not os.path.exists(log_path):
        return None

    metric_value = None
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric")
            if metric == metric_name:
                try:
                    # Pega a última ocorrência (geralmente o resultado final da avaliação)
                    metric_value = float(row["value"])
                except (ValueError, TypeError):
                    continue

    return metric_value


def _plotar_barras_agrupadas(
        modelos: list[str],
        dados: dict[str, list[float]],  # {'metrica1': [v1, v2, ...], 'metrica2': [v1, v2, ...]}
        titulo: str,
        ylabel: str,
        output_filename: str,
        base_results_dir: str,
        limite_y: float | None = None
) -> None:
    """Função genérica para plotar barras agrupadas."""

    METRICAS = list(dados.keys())
    num_modelos = len(modelos)
    num_metricas = len(METRICAS)

    x = np.arange(num_modelos)
    width = 0.8 / max(1, num_metricas)

    # Cores personalizadas
    cores = ['cornflowerblue', 'darkorange', '#4daf4a', '#e41a1c', '#984ea3', '#ff7f00']

    fig, ax = plt.subplots(figsize=(max(10, num_modelos * 1.5), 7))
    retangulos = []

    # Plota as métricas, deslocando o centro de cada barra
    for i, metrica in enumerate(METRICAS):
        valores = dados[metrica]
        # Calcula a posição central para o grupo de barras
        offset = x + (i - (num_metricas - 1) / 2) * width

        rects = ax.bar(offset, valores, width,
                       label=metrica.replace('_', ' ').title(),
                       color=cores[i % len(cores)])
        retangulos.append(rects)

    # --- Configurações Comuns do Gráfico ---
    ax.set_title(titulo, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha="right", fontsize=10)
    ax.legend(shadow=True, loc='best', fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    if limite_y is not None:
        ax.set_ylim(0, limite_y)

    # Função auxiliar para colocar rótulos nas barras
    def autolabel(rects_list):
        formato = '.3f' if limite_y == 1.05 else '.2f'
        for rects in rects_list:
            for rect in rects:
                height = float(rect.get_height())
                if height > 0.01:
                    ax.annotate(f"{height.__format__(formato) + ('s' if 'runtime' in output_filename else '')}",
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=6)

    autolabel(retangulos)

    fig.tight_layout()

    # Define o caminho de saída e salva
    output_dir = Path(base_results_dir) / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGráfico de {titulo.lower()} salvo em: {output_path}")


def comparar_runtimes_modelos(
        specs_datasets: list[tuple[dict, int]],
        base_results_dir: str | Path
):
    METRICAS_RUNTIME = {"Train Runtime (s)": ("train_metrics.csv", "train_runtime"),
                        "Evaluation Runtime (s)": ("eval_metrics.csv", "evaluation_runtime")}

    modelos = []
    dados_plot = {nome_plot: [] for nome_plot in METRICAS_RUNTIME}

    for spec, dataset_num in specs_datasets:
        model_name = spec['name']
        model_label = f"{model_name}-D{dataset_num}"
        modelos.append(model_label)

        log_dir = Path(base_results_dir) / model_name / f"dataset{dataset_num}" / "metrics"

        for nome_plot, (csv_file, metric_name) in METRICAS_RUNTIME.items():
            log_path = log_dir / csv_file
            valor = extrair_metrica_final(log_path, metric_name)
            dados_plot[nome_plot].append(valor if valor is not None else 0.0)

    _plotar_barras_agrupadas(
        modelos=modelos,
        dados=dados_plot,
        titulo='Comparação de Runtimes de Treinamento e Avaliação',
        ylabel='Tempo (segundos)',
        output_filename="runtimes_comparison.svg",
        base_results_dir=base_results_dir,
        limite_y=None
    )


def comparar_metricas_finais(
        specs_datasets: list[tuple[dict, int]],
        base_results_dir: str | Path
):
    METRICAS_FICAIS = ["accuracy", "f1", "precision", "recall"]

    modelos = []
    dados_plot = {m.capitalize(): [] for m in METRICAS_FICAIS}

    for spec, dataset_num in specs_datasets:
        model_name = spec['name']
        model_label = f"{model_name}-D{dataset_num}"
        modelos.append(model_label)

        log_dir = Path(base_results_dir) / model_name / f"dataset{dataset_num}" / "metrics"
        eval_log = log_dir / "eval_metrics.csv"

        for metrica in METRICAS_FICAIS:
            valor = extrair_metrica_final(eval_log, f"eval_{metrica}")
            dados_plot[metrica.capitalize()].append(valor if valor is not None else 0.0)

    if not modelos:
        print("Aviso: Nenhuma métrica de avaliação encontrada para plotagem.")
        return

    _plotar_barras_agrupadas(
        modelos=modelos,
        dados=dados_plot,
        titulo='Comparação de Métricas Finais de Avaliação',
        ylabel='Valor da Métrica (0.0 a 1.0)',
        output_filename="final_metrics_comparison.svg",
        base_results_dir=base_results_dir,
        limite_y=1.05  # Força o limite Y de 0 a 1
    )


if __name__ == "__main__":
    from treinamento import model_specs, dataset_nums

    base_results_dir = Path("./results")
    todas_combinacoes = []

    for spec in model_specs:
        for dataset_num in dataset_nums:
            todas_combinacoes.append((spec, dataset_num))

            plotar_graficos_from_logs(
                output_dir=base_results_dir / spec['name'] / f"dataset{dataset_num}",
                model_name=spec["name"]
            )

    comparar_runtimes_modelos(
        specs_datasets=todas_combinacoes,
        base_results_dir=base_results_dir
    )

    comparar_metricas_finais(
        specs_datasets=todas_combinacoes,
        base_results_dir=base_results_dir
    )
