import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

results_dir = Path("./results")
plots_dir = Path("./plots")
tabelas_dir = Path("./tabelas")

plots_dir.mkdir(exist_ok=True)
tabelas_dir.mkdir(exist_ok=True)

Path("./plots/lr").mkdir(exist_ok=True)
Path("./plots/agrupadas").mkdir(exist_ok=True)

# Estilo único para todo o módulo
sns.set_theme(
    style="whitegrid",
    context="paper"
)

METRIC_TRANSLATION = {
    "recall": "Recall",
    "precision": "Precisão",
    "f1": "F1-Score",
    "accuracy": "Acurácia",
    "eval_accuracy": "Acurácia",
    "eval_f1": "F1-Score",
    "eval_precision": "Precisão",
    "eval_recall": "Recall"
}

tech_order = ["Ajuste Fino", "Transfer Learning", "Do Zero", "Outra"]


# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
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

    return pd.concat(registros, axis=0, ignore_index=True)


def corrigir_runtime_cumulativo(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Devido ao uso de checkpoints para tornar a execução do laço de treino mais eficiente,
     o tempo de execução não é armazenado integralmente, mas sim incrementalmente.
    '''
    df_corr = df.copy()

    # somente runtimes
    mask = df_corr["metric_name"].isin(["train_runtime", "eval_runtime"])
    df_rt = df_corr[mask]

    # ordenar para somar corretamente
    df_rt = df_rt.sort_values(by=["arch", "tech", "dataset", "lr", "metric_name", "epochs"])

    # soma cumulativa por grupo
    df_rt["metric_value"] = (
        df_rt.groupby(["arch", "tech", "dataset", "lr", "metric_name"])["metric_value"]
        .cumsum()
    )

    # substituir valores na base original
    df_corr.loc[mask, "metric_value"] = df_rt["metric_value"]

    return df_corr


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
    s = spec_name.lower().replace("_", " ").replace("-", " ")

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


# ------------------------------------------------------------
# Funções de plotagem
# ------------------------------------------------------------
def plot_curva_loss(df, dataset, lr):
    df_plot = df[
        (df["dataset"] == dataset) &
        (df["lr"] == lr) &
        (df["metric_name"] == "loss") &
        (df["set"] == "train")
        ]

    if df_plot.empty:
        print("Nenhum dado encontrado para os parâmetros fornecidos.")
        return

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_plot,
        x="step",
        y="metric_value",
        hue="spec",
        linewidth=2
    )

    plt.title(f"Curva da Loss (Treino) — Dataset {dataset}, LR={lr}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(plots_dir / "loss" / f"curva_loss-d{dataset}_lr{lr}.svg", bbox_inches="tight")
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
    plt.legend(title="Arquitetura e Técnica de Treinamento", loc='lower center', bbox_to_anchor=(0.4, -0.4), ncol=2)
    plt.ylabel(METRIC_TRANSLATION[metric])
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "lr" / f"lr_vs_{metric}-d{dataset}_e{epochs}.svg", bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Funções de plot agrupado
# ------------------------------------------------------------
def _plot_agrupado_base(
        df_sel: pd.DataFrame,
        datasets,
        title: str,
        ylabel: str,
        out_path: Path,
        value_format=":.3f",
        ylim=None
):
    if df_sel.empty:
        print("Nenhum dado encontrado.")
        return

    archs = sorted(df_sel["arch"].unique())
    techs = [t for t in tech_order if t in df_sel["tech"].unique()] + \
            sorted([t for t in df_sel["tech"].unique() if t not in tech_order])

    n_arch = len(archs)
    n_tech = len(techs)

    width_ds_block = 1.6
    width_arch_block = width_ds_block / n_arch
    bar_width = width_arch_block / n_tech
    arch_spacing = 0.12

    palette = sns.color_palette("Set2", n_colors=n_tech)
    fig, ax = plt.subplots(figsize=(10, 8))

    xticks, xticklabels = [], []

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

            xticks.append(arch_base + (n_tech - 1) * bar_width / 2)
            xticklabels.append(arch)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha="center")

    if len(datasets) == 1:
        ds_part = f" — Dataset {datasets[0]}"
    else:
        ds_part = " — " + ", ".join([f"Dataset {d}" for d in datasets])

    ax.set_title(title + ds_part)
    ax.set_ylabel(ylabel)

    ax.legend(
        [plt.Line2D([0], [0], color=palette[i], lw=8) for i in range(n_tech)],
        techs,
        title="Técnica",
        loc="upper right",
        framealpha=0.85,
        facecolor="white"
    )

    # Anotar valores
    for rect in ax.patches:
        h = rect.get_height()
        if not np.isnan(h):
            ax.annotate(
                format(h, value_format),
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9
            )

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plotar_metricas_agrupadas_por_modelo(df, metric_name, lr, epochs, datasets=(1, 2)):
    df_sel = df[
        (df["lr"] == float(lr)) &
        (df["epochs"] == int(epochs)) &
        (df["metric_name"] == metric_name)
        ].copy()

    title = f"{METRIC_TRANSLATION[metric_name]} — LR={lr}, Épocas={epochs}"
    ylabel = METRIC_TRANSLATION[metric_name]
    out_path = plots_dir / "agrupadas" / f"comparacao_{metric_name}-lr{lr}_e{epochs}.svg"

    _plot_agrupado_base(
        df_sel,
        datasets,
        title,
        ylabel,
        out_path,
        value_format=".3f",
        ylim=(0, 1.05)
    )


def plotar_runtime_medio_por_dataset_epochs(
        df_final: pd.DataFrame,
        dataset: int,
        epochs: int,
        modo="train"  # "eval" ou "train"
):
    metric_name = f"{modo}_runtime"

    # Seleciona e faz média sobre LR
    df_sel = (
        df_final[
            (df_final["dataset"] == int(dataset)) &
            (df_final["epochs"] == int(epochs)) &
            (df_final["metric_name"] == metric_name)
            ]
        .groupby(["arch", "tech"], as_index=False)["metric_value"].mean()
    )

    if df_sel.empty:
        print(f"Sem runtime para dataset={dataset}, epochs={epochs}.")
        return

    # Para reutilizar a função base, precisamos introduzir um campo "dataset" artificial,
    # pois ela espera múltiplos datasets (mesmo se for só um).
    df_sel["dataset"] = dataset
    datasets = (dataset,)  # tupla de um elemento

    title = f"Tempo de {'Treinamento' if modo == 'train' else 'Avaliação'} Médio — Épocas={epochs}"
    ylabel = "Tempo médio (segundos)"
    out_path = plots_dir / "runtime_medio" / f"{modo}_d{dataset}_e{epochs}.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _plot_agrupado_base(
        df_sel,
        datasets,
        title,
        ylabel,
        out_path,
        value_format=".2f",
        ylim=None
    )


# ------------------------------------------------------------
# Funções de impressão de tabela latex
# ------------------------------------------------------------
def format_lr_pow10(lr: float) -> str:
    """
    Formata o learning rate como potência de 10 para LaTeX.
    Ex: 0.001 -> '10^{-3}', 5e-4 -> '5×10^{-4}'
    """
    # Converte para notação científica
    s = f"{lr:.1e}"  # ex: '5.0e-04'
    mantissa, exp = s.split("e")
    mantissa = float(mantissa)
    exp = int(exp)

    # Caso seja 1 × 10^n, omitimos o 1
    if mantissa == 1:
        return f"$10^{{{exp}}}$"

    return f"${mantissa} \\times 10^{{{exp}}}$"


def salvar_tabela_latex(df, metric_name: str, dataset: int, menor_melhor=False):
    df = df.copy()

    df = df[(df["dataset"] == dataset) & (df["metric_name"] == metric_name)]
    if df.empty:
        return

    df = df.sort_values(by=["arch", "tech", "epochs", "lr"])
    df["col_label"] = df.apply(
        lambda x: f"{format_lr_pow10(x['lr'])}\\textbar{x['epochs']}",
        axis=1
    )

    modelos = df["arch"].unique()

    for modelo in modelos:
        df_m = df[df["arch"] == modelo]

        tabela = df_m.pivot_table(index="tech", columns="col_label", values="metric_value")
        tabela = tabela.reindex(index=[t for t in tech_order if t in tabela.index])

        # Seleção de melhores valores
        if menor_melhor:
            best_by_row = tabela.min(axis=1)
            best_by_col = tabela.min(axis=0)
        else:
            best_by_row = tabela.max(axis=1)
            best_by_col = tabela.max(axis=0)

        tabela_fmt = tabela.copy().astype(object)

        for tech, row in tabela.iterrows():
            for col in tabela.columns:
                val = row[col]
                if pd.isna(val):
                    tabela_fmt.loc[tech, col] = "--"
                    continue

                s = f"{float(val):.3f}"

                if val == best_by_row.loc[tech]:
                    s = f"\\textbf{{{s}}}"
                if val == best_by_col[col]:
                    s = f"\\textit{{{s}}}"

                tabela_fmt.loc[tech, col] = s

        linhas = [
            f"{tech} & " + " & ".join(str(tabela_fmt.loc[tech, c]) for c in tabela_fmt.columns) + " \\\\"
            for tech in tabela_fmt.index
        ]

        col_header = " & ".join(tabela_fmt.columns)

        unidade = "Tempo (s)" if "runtime" in metric_name else METRIC_TRANSLATION.get(metric_name, metric_name)

        latex = (
                f"\\begin{{table}}[htbp]\n"
                f"\\scriptsize\n"
                f"\\centering\n"
                f"\\caption{{{unidade} no \\textit{{Dataset}} {dataset}, Modelo {modelo}}}\n"
                f"\\label{{tab:{metric_name}_d{dataset}_{modelo}}}\n"
                f"\\begin{{tabularx}}{{\\linewidth}}{{X {'X ' * len(tabela_fmt.columns)}}}\n"
                f"\\hline\n"
                f"\\multirow{{2}}{{*}}{{Técnica}} & \\multicolumn{{9}}{{c}}{{Learning Rate | Número de Épocas}} \\\\\n"
                f"\\hhline{{~---------}}\n"
                f" & {col_header} \\\\\n"
                f"\\hline\n" +
                "\n".join(linhas) +
                "\n\\hline\n"
                "\\end{tabularx}\\\\\n"
                "\\vspace{0.3cm}\n"
                "\\small Fonte: elaboração própria\n"
                "\\end{table}\n"
        )

        (tabelas_dir / f"tabela_{metric_name}_d{dataset}_{modelo}.tex").write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    df = corrigir_runtime_cumulativo(
        extrair_metricas_finais(
            carregar_metricas_formatado()
        )
    )

    # Lista de métricas consideradas relevantes para comparação
    metricas_interesse = [
        "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"
    ]

    for metric in metricas_interesse:
        # Seleciona apenas linhas dessa métrica
        df_metric = df[df["metric_name"] == metric]
        if df_metric.empty:
            continue

        # Itera por combinações únicas de learning rate e epochs
        for dataset in sorted(df_metric["dataset"].unique()):
            salvar_tabela_latex(
                df=df,
                metric_name=metric,
                dataset=dataset,
            )

            salvar_tabela_latex(
                df=df,
                metric_name="train_runtime",
                dataset=dataset,
                menor_melhor=True
            )

            for epochs in sorted(df_metric["epochs"].unique()):
                plot_lr_vs_metric(
                    dataset=dataset,
                    df=df_metric,
                    metric=metric,
                    epochs=epochs,
                )

                plotar_runtime_medio_por_dataset_epochs(
                    df_final=df,
                    dataset=dataset,
                    epochs=epochs,
                    modo="train"
                )

                for lr in sorted(df_metric["lr"].unique()):
                    plotar_metricas_agrupadas_por_modelo(
                        df=df,
                        metric_name=metric,
                        lr=lr,
                        epochs=epochs,
                    )
