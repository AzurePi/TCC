import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator


def useless_metric(metric: str):
    useless = [
        "total_flos",
        "train_loss",
        "learning_rate",
        "grad_norm",
        "model_preparation_time"
    ]

    useless_begginings = (
        'epoch'
    )

    useless_endings = (
        'runtime',
        'per_second'
    )

    if metric in useless or metric.startswith(useless_begginings) or metric.endswith(useless_endings):
        return True
    return False


def carregar_logs_csv(train_log_path: Path, eval_log_path: Path):
    """Carrega métricas de treino, validação e baseline a partir dos CSVs do callback."""
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


def grafico_aux(
        titulo: str,
        treino: list[float] | None,
        validacao: list[float] | None,
        ylabel: str,
        output_path: Path
) -> None:
    """Plota uma curva, podendo ser treino, validação, ou ambos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(titulo)

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


def plotar_graficos_from_logs(exec_dir: str, model_name: str):
    """
    Lê os logs CSV do callback e gera gráficos de perda e acurácia.

    :param exec_dir: diretório onde estão 'train_metrics.csv' e 'eval_metrics.csv'
    :param model_name: usado para nomear os arquivos de saída
    """
    exec_dir = Path(exec_dir)

    train_log = exec_dir / "train_metrics.csv"
    eval_log = exec_dir / "eval_metrics.csv"

    if not train_log.exists():
        print(f"[IGNORADO] Treino não encontrado em {train_log}")
        return

    df_train = pd.read_csv(train_log, names=["step", "metric", "value"], header=0)
    df_eval = pd.read_csv(eval_log, names=["step", "metric", "value"], header=0) if eval_log.exists() else None

    # Gráfico de Loss de treino
    df_loss = df_train[df_train["metric"] == "loss"]
    plt.figure()
    plt.plot(df_loss["step"], df_loss["value"])
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title(f"{model_name} - Train Loss")
    plt.tight_layout()
    plt.show()

    # Gráfico de acurácia de validação, se existir
    if df_eval is not None:
        df_acc = df_eval[df_eval["metric"] == "eval_accuracy"]
        if not df_acc.empty:
            plt.figure()
            plt.plot(df_acc["step"], df_acc["value"])
            plt.xlabel("Step")
            plt.ylabel("Eval Accuracy")
            plt.title(f"{model_name} - Eval Accuracy")
            plt.tight_layout()
            plt.show()


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


def plotar_barras_agrupadas(
        dados: dict,
        titulo: str,
        ylabel: str,
        output_path: Path,
        ylim: tuple[float, float] | None = None,
        formato_label: str = "{:.2f}"
):
    """
    Plota gráfico de barras agrupadas (Scratch vs Baseline vs FT).
    """
    chaves_ordenadas = sorted(dados.keys(), key=lambda x: (x[0], x[1]))

    labels = [f"{modelo}-D{ds}" for ds, modelo in chaves_ordenadas]
    grupos = sorted({g for v in dados.values() for g in v.keys()})  # ex: ["Baseline", "Scratch", "FT"]

    x = np.arange(len(labels))
    width = 0.8 / len(grupos)  # largura dinâmica

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 7))

    rects = []
    for i, grupo in enumerate(grupos):
        valores = [dados[chave].get(grupo, 0.0) for chave in chaves_ordenadas]
        rects.append(
            ax.bar(x + (i - (len(grupos) - 1) / 2) * width, valores, width, label=grupo)
        )

    ax.set_title(titulo, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(shadow=True, loc="best", fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    def autolabel(rects):
        for rect_group in rects:
            for rect in rect_group:
                height = float(rect.get_height())
                if height > 0.01:
                    ax.annotate(formato_label.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center", va="bottom", fontsize=6)

    autolabel(rects)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {output_path}")


def comparar_runtimes_modelos(resultados_dir: str, modelos: list, dataset_num: int):
    dados = []

    for model_name in modelos:
        base_dir = Path(resultados_dir) / model_name / f"dataset{dataset_num}"
        exec_dirs = base_dir.glob("lr_*_epochs_*")

        for d in exec_dirs:
            metrics_path = d / "eval_metrics.csv"
            if not metrics_path.exists():
                continue

            df = pd.read_csv(metrics_path, names=["step", "metric", "value"], header=0)
            tempo = df[df["metric"] == "tempo_total_segundos"]

            if len(tempo) == 0:
                continue

            tempo_final = float(tempo["value"].iloc[-1])
            dados.append((model_name, d.name, tempo_final))

    if not dados:
        print("Nenhum runtime registrado.")
        return

    df_time = pd.DataFrame(dados, columns=["modelo", "execucao", "tempo_segundos"])
    print(df_time)

    plt.figure()
    df_time.groupby("modelo")["tempo_segundos"].mean().plot(kind="bar")
    plt.ylabel("Tempo médio (s)")
    plt.title("Comparação de Tempo Médio de Treino")
    plt.tight_layout()
    plt.show()


def comparar_metricas_finais(resultados_dir: str, model_name: str, dataset_num: int):
    base_dir = Path(resultados_dir) / model_name / f"dataset{dataset_num}"
    exec_dirs = sorted(base_dir.glob("lr_*_epochs_*"))

    resultados = []

    for d in exec_dirs:
        eval_log = d / "eval_metrics.csv"
        if not eval_log.exists():
            continue

        df = pd.read_csv(eval_log, names=["step", "metric", "value"], header=0)
        df_acc = df[df["metric"] == "eval_accuracy"]
        if df_acc.empty:
            continue

        final_acc = df_acc.iloc[-1]["value"]

        # extrai hiperparâmetros do nome da pasta
        _, lr_val, _, epochs_val = d.name.split("_")
        resultados.append((float(lr_val), int(epochs_val), final_acc))

    if not resultados:
        print("Nenhuma execução encontrada.")
        return

    resultados = pd.DataFrame(resultados, columns=["lr", "epochs", "accuracy"])
    print(resultados)

    plt.figure()
    for ep in sorted(resultados["epochs"].unique()):
        df_ep = resultados[resultados["epochs"] == ep]
        plt.plot(df_ep["lr"], df_ep["accuracy"], marker="o", label=f"{ep} epochs")

    plt.xlabel("Learning Rate")
    plt.ylabel("Final Eval Accuracy")
    plt.title(f"{model_name} - Comparação por LR e Épocas")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_baseline(model_name, dataset_name, results_root):
    f = Path(results_root) / model_name / dataset_name / "baseline_metrics.json"
    if not f.exists():
        return None
    with open(f, "r") as fp:
        data = json.load(fp)
    return data.get("accuracy", None)


def comparar_metricas_finais_por_dataset(resultados_dir: str, model_specs, dataset_num: int):
    dados = {}  # {(tipo_modelo, dataset_num): {estrategia: accuracy}}

    for spec in model_specs:
        tipo = spec["type"]  # CNN ou ViT
        grupo = classificar_treinamento(spec["transfer"], spec["finetuning"])
        base_dir = Path(resultados_dir) / spec["name"] / f"dataset{dataset_num}"
        exec_dirs = sorted(base_dir.glob("lr_*_epochs_*"))

        melhor_acc = None
        for d in exec_dirs:
            eval_log = d / "eval_metrics.csv"
            if not eval_log.exists():
                continue

            df = pd.read_csv(eval_log, names=["step", "metric", "value"], header=0)
            df_acc = df[df["metric"] == "eval_accuracy"]
            if df_acc.empty:
                continue

            final_acc = float(df_acc.iloc[-1]["value"])
            melhor_acc = final_acc if (melhor_acc is None or final_acc > melhor_acc) else melhor_acc

        if melhor_acc is None:
            continue

        chave = (tipo, dataset_num)
        dados.setdefault(chave, {})
        dados[chave][grupo] = melhor_acc

    return dados


def plotar_por_dataset(dados_dataset, dataset_num: int, output_dir: Path):
    for tipo in ["CNN", "ViT"]:
        chave = (tipo, dataset_num)
        if chave not in dados_dataset:
            continue

        grupos = ["Transfer Learning", "Ajuste Fino", "Do Zero"]
        valores = [dados_dataset[chave].get(g, 0.0) for g in grupos]

        plt.figure(figsize=(6, 4))
        plt.bar(grupos, valores)
        plt.title(f"{tipo} - Dataset {dataset_num}")
        plt.ylabel("Acurácia Final")
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", linewidth=0.5)

        for i, v in enumerate(valores):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{tipo}_dataset{dataset_num}.png", bbox_inches="tight")
        plt.close()


def plotar_grupos_por_dataset(dados_dataset, dataset_num: int, output_dir: Path):
    tipos = ["CNN", "ViT"]
    estrategias = ["Transfer Learning", "Ajuste Fino", "Do Zero"]

    # Matriz de valores: rows = tipos, cols = estrategias
    valores = []
    for tipo in tipos:
        chave = (tipo, dataset_num)
        linha = [dados_dataset.get(chave, {}).get(e, 0.0) for e in estrategias]
        valores.append(linha)

    valores = np.array(valores)  # shape: (2, 3)

    x = np.arange(len(tipos))  # [0, 1]
    width = 0.25  # largura de cada barra dentro do grupo

    fig, ax = plt.subplots(figsize=(8, 5))

    # Desenho das barras
    for i, estrategia in enumerate(estrategias):
        ax.bar(x + (i - 1) * width, valores[:, i], width, label=estrategia)

    ax.set_title(f"Comparação de Estratégias — Dataset {dataset_num}", fontsize=14)
    ax.set_ylabel("Acurácia Final")
    ax.set_xticks(x)
    ax.set_xticklabels(tipos, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Estratégias", fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    # Rotular valores numéricos
    for i in range(len(tipos)):
        for j in range(len(estrategias)):
            valor = valores[i, j]
            if valor > 0:
                ax.text(i + (j - 1) * width, valor + 0.01, f"{valor:.3f}",
                        ha="center", va="bottom", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"Comparacao_Estrategias_dataset{dataset_num}.png", bbox_inches="tight")
    plt.close(fig)


def classificar_treinamento(spec):
    if not spec["transfer"]:
        return "Do Zero"
    if spec["transfer"] and not spec["finetuning"]:
        return "Transfer Learning"
    if spec["finetuning"]:
        return "Ajuste Fino"
    return None


if __name__ == "__main__":
    from treinamento import model_specs, dataset_nums, num_epochs, learning_rates

    for n_epochs in num_epochs:
        for lr in learning_rates:

            # Para cada dataset, agrupamos resultados para comparação
            for dataset_num in dataset_nums:

                # Lista de nomes de modelos para comparação
                nomes_modelos = [spec["name"] for spec in model_specs]

                print(f"\n=== Resultados: dataset{dataset_num} | lr={lr} | epochs={n_epochs} ===")

                # 1) Plot para cada modelo individual
                for spec in model_specs:
                    exec_dir = (
                            Path("./results")
                            / spec["name"]
                            / f"dataset{dataset_num}"
                            / f"lr_{lr}_epochs_{n_epochs}"
                    )

                    plotar_graficos_from_logs(
                        exec_dir=str(exec_dir),
                        model_name=f"{spec['name']} (dataset{dataset_num}, lr={lr}, e={n_epochs})"
                    )

                # 2) Comparar métricas finais entre modelos
                dados = comparar_metricas_finais_por_dataset(
                    resultados_dir="./results",
                    model_specs=model_specs,
                    dataset_num=dataset_num
                )

                plotar_grupos_por_dataset(
                    dados_dataset=dados,
                    dataset_num=dataset_num,
                    output_dir=Path("./plots")
                )

                # 3) Comparar tempo de treinamento entre modelos
                comparar_runtimes_modelos(
                    resultados_dir="./results",
                    modelos=nomes_modelos,
                    dataset_num=dataset_num
                )
