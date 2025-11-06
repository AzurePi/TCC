import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

results_dir = Path("./results")
plots_dir = Path("./plots")
plots_dir.mkdir(exist_ok=True)
Path("./plots/lr").mkdir(exist_ok=True)
Path("./plots/agrupadas").mkdir(exist_ok=True)
Path("./plots/latex").mkdir(exist_ok=True)
#Path("./plots/loss").mkdir(exist_ok=True)

# Estilo único para todo o módulo
sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.2
)

METRIC_TRANSLATION = {
    "recall": "Recall",
    "precision": "Precisão",
    "f1": "F1-Score",
    "accuracy": "Precisão",
    "eval_accuracy": "Precisão",
    "eval_f1": "F1-Score",
    "eval_precision": "Precisão",
    "eval_recall": "Recall"
}


def carregar_metricas():
    registros = []

    for spec_dir in results_dir.iterdir():
        if not spec_dir.is_dir():
            continue

        spec_name = spec_dir.name

        for dataset_dir in spec_dir.iterdir():
            if not dataset_dir.is_dir() or not dataset_dir.name.startswith("d"):
                continue

            dataset = int(dataset_dir.name[1:])  # d1 → 1

            for combo_dir in dataset_dir.iterdir():
                if not combo_dir.is_dir() or not combo_dir.name.startswith("lr_"):
                    continue

                # Parseia lr e epochs
                # exemplo: lr_0.001_e_3
                _, lr, _, epochs = combo_dir.name.split("_")
                lr = float(lr)
                epochs = int(epochs)

                # Arquivos
                train_file = combo_dir / "train_metrics.csv"
                eval_file = combo_dir / "eval_metrics.csv"

                if train_file.exists():
                    df_train = pd.read_csv(train_file).assign(
                        set="train",
                        spec=spec_name,
                        dataset=dataset,
                        lr=lr,
                        epochs=epochs
                    )
                    registros.append(df_train)

                if eval_file.exists():
                    df_eval = pd.read_csv(eval_file).assign(
                        set="eval",
                        spec=spec_name,
                        dataset=dataset,
                        lr=lr,
                        epochs=epochs
                    )
                    registros.append(df_eval)

    return pd.concat(registros, axis=0, ignore_index=True, copy=False)


def carregar_metricas_formatado():
    df = carregar_metricas()

    # padroniza nomes
    df = df.rename(columns={"metric": "metric_name", "value": "metric_value"})

    def safe_split(s):
        a, t = split_spec_to_arch_and_tech(s)
        return pd.Series([a, t])

    # separa arquitetura e técnica
    df[["arch", "tech"]] = df["spec"].apply(safe_split)

    return df


def extrair_metricas_finais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada combinação (spec, dataset, lr, epochs, set, metric) retorna a linha
    correspondente ao maior `step` (assume que é a métrica 'final' guardada).
    """
    if "step" not in df.columns:
        df = df.assign(step=0)
    return df.loc[df.groupby(["spec", "dataset", "lr", "epochs", "set", "metric_name"])["step"].idxmax()]


def split_spec_to_arch_and_tech(spec_name: str):
    """
    Retorna (architecture, technique) a partir de spec_name.
    Expectativa sobre spec_name baseada em exemplos:
    "ConvNext - Ajuste Fino", "Visual Transformer - Do Zero", etc.
    """
    s = spec_name.lower()
    # Arquitetura
    if "convnext" in s:
        arch = "ConvNext"
    elif "vit" in s or "visual transformer" in s:
        arch = "ViT"
    else:
        # fallback: pega primeira palavra capitalizada
        arch = spec_name.split()[0]

    # Técnica
    if "transfer" in s and "fin" not in s:  # "Transfer Learning" mas não "Ajuste Fino"
        tech = "Transfer Learning"
    elif "finetun" in s or "ajuste fino" in s or "ajuste" in s:
        tech = "Ajuste Fino"
    elif "zero" in s or "do zero" in s:
        tech = "Do Zero"
    else:
        # tenta detectar "transfer" substring geral
        if "transfer" in s:
            tech = "Transfer Learning"
        else:
            tech = "Outra"

    return arch, tech


def plot_curva_loss(df, dataset, lr):
    df_plot = df[
        (df["dataset"] == dataset) &
        (df["lr"] == lr) &
        (df["metric"] == "loss") &
        (df["set"] == "train")
        ]

    if df_plot.empty:
        print("Nenhum dado encontrado para os parâmetros fornecidos.")
        return

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_plot,
        x="step",
        y="value",
        hue="spec",
        linewidth=2
    )

    plt.title(f"Curva da Loss (Treino) — Dataset {dataset}, LR={lr}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(plots_dir / "loss" / f"/curva_loss-d{dataset}_lr{lr}.png", bbox_inches="tight")
    plt.close()


def plot_lr_vs_metric(df, dataset, epochs, metric="eval_accuracy"):
    df_plot = df[
        (df["dataset"] == dataset) &
        (df["epochs"] == epochs) &
        (df["metric_name"] == metric) &
        (df["set"] == "eval")
        ]

    if df_plot.empty:
        print("Nenhum dado encontrado para os parâmetros fornecidos.")
        return

    # Ordena para plot mais limpo
    df_plot = df_plot.sort_values(by="lr")

    plt.figure(figsize=(6, 8))
    sns.lineplot(
        data=df_plot,
        x="lr",
        y="metric_value",
        hue="spec",
        marker="o",
        linewidth=2,
        errorbar=None
    )

    plt.xscale("log")  # LR é melhor interpretada em escala log
    plt.title(f"{METRIC_TRANSLATION[metric]} vs Learning Rate — Dataset {dataset}, Epochs={epochs}")
    plt.xlabel("Learning Rate")
    plt.legend(title="Arquitetura e Técnica de Treinamento", loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.ylabel(METRIC_TRANSLATION[metric])
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "lr" / f"lr_vs_{metric}-d{dataset}_e{epochs}.png", bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Função principal: plot agrupado
# ------------------------------------------------------------
def plotar_metricas_agrupadas_por_modelo(
        df_final: pd.DataFrame,
        metric_name: str,
        lr: float,
        epochs: int,
        datasets=(1, 2, 3),
):
    df_sel = df_final[
        (df_final["lr"] == float(lr)) &
        (df_final["epochs"] == int(epochs)) &
        (df_final["metric_name"] == metric_name)
        ].copy()

    if df_sel.empty:
        print("Nenhum dado encontrado.")
        return

    archs = sorted(df_sel["arch"].unique())
    tech_order = ["Ajuste Fino", "Transfer Learning", "Do Zero", "Outra"]
    techs = [t for t in tech_order if t in df_sel["tech"].unique()] + \
            sorted([t for t in df_sel["tech"].unique() if t not in tech_order])

    n_arch = len(archs)
    n_tech = len(techs)

    # a hierarquia é: DATASET → ARQUITETURA → TÉCNICA
    width_ds_block = 1.6
    width_arch_block = width_ds_block / n_arch
    bar_width = width_arch_block / n_tech
    arch_spacing = 0.12  # deslocamento adicional entre arquiteturas

    palette = sns.color_palette("Set2", n_colors=n_tech)

    fig, ax = plt.subplots(figsize=(10, 8))

    xticks = []
    xticklabels = []

    for i_ds, ds in enumerate(datasets):
        ds_base = i_ds * (width_ds_block + 0.6)

        for i_arch, arch in enumerate(archs):
            arch_base = ds_base + i_arch * (width_arch_block + arch_spacing)

            for i_tech, tech in enumerate(techs):
                xpos = arch_base + i_tech * bar_width

                row = df_sel[(df_sel["dataset"] == ds) &
                             (df_sel["arch"] == arch) &
                             (df_sel["tech"] == tech)]
                val = float(row.iloc[0]["metric_value"]) if not row.empty else np.nan

                ax.bar(
                    xpos,
                    val,
                    bar_width * 0.9,
                    color=palette[i_tech],
                    edgecolor="black",
                    alpha=0.9
                )

            # Tick central para arquitetura
            xticks.append(arch_base + (n_tech - 1) * bar_width / 2)
            xticklabels.append(arch)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha="center")

    # Adiciona uma label por dataset abaixo do grupo de arquiteturas
    for i_ds, ds in enumerate(datasets):
        ds_base = i_ds * (width_ds_block + 0.6)
        center = ds_base + (width_ds_block - bar_width + arch_spacing) / 2
        ax.text(
            center,  # posição x
            -0.08,  # posição y (fração do eixo)
            f"d{ds}",
            ha="center", va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=11
        )

    ax.set_title(f"{METRIC_TRANSLATION[metric_name]} — LR={lr}, Épocas={epochs}")
    ax.set_ylabel(METRIC_TRANSLATION[metric_name])

    ax.legend(
        [plt.Line2D([0], [0], color=palette[i], lw=8) for i in range(n_tech)],
        techs, title="Técnica de Treinamento",
        loc='lower center', bbox_to_anchor=(0.5, -0.3), ncols=3
    )

    # Anota valores
    for rect in ax.patches:
        h = rect.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(plots_dir / "agrupadas" / f"comparacao_{metric_name}-lr{lr}_e{epochs}.png", dpi=200)
    plt.close()


def salvar_tabela_latex(df, metric_name: str, output_dir: Path = "./"):
    """
    df deve conter: arch, technique, epochs, lr, metric_value
    Salva a tabela LaTeX em output_dir/tabela_{metric_name}.tex
    """
    df = df.copy()
    df = df.sort_values(by=["arch", "tech", "epochs", "lr"])

    df["col_label"] = df.apply(lambda x: f"{x['lr']}/{x['epochs']}", axis=1)
    tabela = df.pivot_table(index=["arch", "tech"], columns="col_label", values="metric_value")

    best_by_row = tabela.max(axis=1)
    best_by_col = tabela.max(axis=0)

    tabela_fmt = tabela.copy().astype(object)
    for (m, t), row in tabela.iterrows():
        for col in tabela.columns:
            val = row[col]
            if pd.isna(val):
                continue
            s = f"{val:.4f}"
            if val == best_by_row.loc[(m, t)]:
                s = f"\\textbf{{{s}}}"
            if val == best_by_col[col]:
                s = f"\\textit{{{s}}}"
            tabela_fmt.loc[(m, t), col] = s

    linhas = []
    last_model = None
    for (model, technique), row in tabela_fmt.iterrows():
        if model != last_model:
            count = (tabela_fmt.index.get_level_values(0) == model).sum()
            linhas.append(f"\\multirow{{{count}}}{{*}}{{{model}}} & {technique}")
            last_model = model
        else:
            linhas.append(f" & {technique}")

        linha_vals = " & ".join(str(row[c]) for c in tabela_fmt.columns)
        linhas[-1] += " & " + linha_vals + " \\\\"

    col_header = " & ".join(tabela_fmt.columns)

    latex = (
            f"\\begin{{table}}[htbp]\n"
            f"\\centering\n"
            f"\\caption{{Resultados da métrica {METRIC_TRANSLATION[metric_name]}}}\n"
            f"\\label{{tab:{METRIC_TRANSLATION[metric_name]}}}\n"
            f"\\begin{{tabularx}}{{X X {'X ' * len(tabela_fmt.columns)}}}\n"
            f"\\hline\n"
            f"Modelo & Técnica & {col_header} \\\\\n"
            f"\\hline\n" +
            "\n".join(linhas) +
            "\n\\hline\n"
            "\\end{tabularx}\n"
            "\\small Fonte: Elaboração própria\n"
            "\\end{table}\n"
    )

    file_path = f"{output_dir}/tabela_{metric_name}.tex"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(latex)


if __name__ == "__main__":
    df = carregar_metricas_formatado()

    # Lista de métricas consideradas relevantes para comparação
    metricas_interesse = [
        "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"
    ]

    for metric in metricas_interesse:
        salvar_tabela_latex(
            df=df,
            metric_name=metric,
            output_dir=plots_dir / "latex"
        )

        # Seleciona apenas linhas dessa métrica
        df_metric = df[df["metric_name"] == metric]
        if df_metric.empty:
            continue

        # Itera por combinações únicas de learning rate e epochs
        for epochs in sorted(df_metric["epochs"].unique()):

            for dataset in sorted(df_metric["dataset"].unique()):
                plot_lr_vs_metric(
                    dataset=dataset,
                    df=df_metric,
                    metric=metric,
                    epochs=epochs,
                )

            for lr in sorted(df_metric["lr"].unique()):

                df_sel = df_metric[
                    (df_metric["lr"] == lr) &
                    (df_metric["epochs"] == epochs)
                    ]

                if df_sel.empty:
                    continue

                plotar_metricas_agrupadas_por_modelo(
                    df_final=df_sel,
                    metric_name=metric,
                    lr=lr,
                    epochs=epochs,
                )
