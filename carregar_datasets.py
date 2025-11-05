import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kaggle
import numpy as np
from datasets import load_dataset, DatasetDict
from matplotlib import pyplot as plt
from transformers import BaseImageProcessor


def carregar_datasets(
        dataset_num: int,
        validation_split: float,
        processor: BaseImageProcessor,
        amostrar: bool = False,
) -> tuple[DatasetDict, list[str]]:
    """
    Carrega e prepara datasets de treinamento, validação e teste para HuggingFace Trainer.
    """
    # Carrega conjuntos de imagens
    print(f"Carregando dataset {dataset_num}...")
    raw_datasets = DatasetDict({
        'train': load_dataset("imagefolder", data_dir=f"./treinamento{dataset_num}", split="train"),
        'test': load_dataset("imagefolder", data_dir=f"./teste{dataset_num}", split="train"),
    })

    # Split em treino, teste e validação
    split = raw_datasets["train"].train_test_split(test_size=validation_split, seed=42)
    raw_datasets["train"] = split["train"]
    raw_datasets["validation"] = split["test"]

    class_names = raw_datasets["train"].features["label"].names

    # Pré-processamento
    print("Pré-processamento...")

    def transform(examples):
        if "image" in examples:
            images = [img.convert("RGB") for img in examples["image"]]
            inputs = processor(images)
            inputs["labels"] = examples["label"]
            return inputs
        else:
            raise ValueError(f"Coluna 'image' não está presente. Keys recebidas: {examples.keys()}")

    datasets = raw_datasets.map(transform, batched=True, num_proc=8, remove_columns=["image", "label"])
    datasets.set_format(type="torch", columns=["pixel_values", "labels"])

    if amostrar:
        salvar_amostras(raw_datasets["train"], dataset_num)
        salvar_amostras(datasets["train"], dataset_num)
    return datasets, class_names


def preparar_diretorios(test_split: float, dataset_dirs=None) -> list[list[int]]:
    """
    Prepara os diretórios e conjuntos de dados para treinamento e teste.

    Verifica se os diretórios de datasets específicos existem. Se não, prepara os datasets necessários e
    cria os diretórios de treinamento e teste dividindo as imagens conforme a proporção especificada para teste.
    Faz uma combinação dos Datasets 1 e 2 para criar o Dataset 3.

    :param test_split: Proporção dos dados a serem utilizados para teste.
    :param dataset_dirs: Lista dos caminhos para os diretórios de imagens.
    :return: Uma lista contendo sublistas com o número de exemplos positivos e negativos para cada dataset.
    """
    if dataset_dirs is None:
        dataset_dirs = [Path("./dataset1"), Path("./dataset2")]
    N = []

    # 1. Prepara Datasets 1 e 2
    for i, dataset_dir in enumerate(dataset_dirs, start=1):
        if i == 1:
            preparar_dataset1(dataset_dir)
        elif i == 2:
            preparar_dataset2(dataset_dir)

        num_positivas = len(list((dataset_dir / "positivo").glob("*")))
        num_negativas = len(list((dataset_dir / "negativo").glob("*")))
        treinamento_e_teste(i, num_positivas, num_negativas, test_split, dataset_dir)
        N.append([num_positivas, num_negativas])

    # 2. Combina Datasets 1 e 2 para criar o Dataset 3
    combinar_datasets()

    # 3. Conta o número de exemplos para o Dataset 3 e adiciona a N
    treino3_dir = Path("treinamento3")
    if treino3_dir.exists():
        num_pos_treino3 = len(list((treino3_dir / "positivo").glob("*")))
        num_neg_treino3 = len(list((treino3_dir / "negativo").glob("*")))

        teste3_dir = Path("teste3")
        num_pos_teste3 = len(list((teste3_dir / "positivo").glob("*")))
        num_neg_teste3 = len(list((teste3_dir / "negativo").glob("*")))

        total_pos = num_pos_treino3 + num_pos_teste3
        total_neg = num_neg_treino3 + num_neg_teste3

        N.append([total_pos, total_neg])

    return N


def formatar_diretorio(origem: Path, destino: Path) -> None:
    """
    Move todos os arquivos de um diretório de origem para um destino e remove o diretório de origem.

    :param origem: Diretório de origem contendo os arquivos.
    :param destino: Diretório de destino.
    :return: None
    """

    def _move_single_file(file: Path, destino: Path) -> None:
        """Função auxiliar para mover um único arquivo, usada pelo ThreadPoolExecutor."""
        shutil.move(file, destino / file.name)

    destino.mkdir(parents=True, exist_ok=True)

    arquivos_a_mover = [file for file in origem.iterdir() if file.is_file()]

    if not arquivos_a_mover:
        print("\tNenhum arquivo para mover. Removendo diretório de origem vazio.")
        shutil.rmtree(origem, ignore_errors=True)
        return

    # 1. Paraleliza a movimentação dos arquivos
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Mapeia a função _move_single_file para cada arquivo, passando o destino como argumento fixo
        futures = [executor.submit(_move_single_file, file, destino) for file in arquivos_a_mover]

        # Espera que todas as movimentações terminem e verifica por exceções
        for future in futures:
            future.result()

    # 2. Remove o diretório de origem após a conclusão de todas as movimentações
    shutil.rmtree(origem)


def preparar_dataset1(dataset_dir: Path = Path("./dataset1")) -> None:
    """
    Prepara o ambiente com o dataset "`sarscov2-ctscan-dataset`", baixado do kaggle.

    Se o dataset ainda não foi baixado, baixamos e descompactamos.

    :param dataset_dir: Um diretório onde o dataset será armazenado.
    :return: None
    """
    if not dataset_dir.exists():
        print("Baixando dataset 1 de imagens e criando diretório...")
        dataset_dir.mkdir()

        kaggle.api.dataset_download_files('plameneduardo/sarscov2-ctscan-dataset', path=dataset_dir, unzip=True)

        positivo_dir = dataset_dir / "COVID"
        negativo_dir = dataset_dir / "non-COVID"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")

        print("Pronto!\n")
    else:
        print("Diretório de imagens para o dataset 1 já está presente na máquina. Prosseguindo...\n")


def preparar_dataset2(dataset_dir: Path = Path("./dataset2"), include_ni=False) -> None:
    """
    Prepara o ambiente com o dataset "`preprocessed-ct-scans-for-covid19`", baixado do kaggle.

    Se o dataset ainda não foi baixado, baixamos e descompactamos. Mantemos apenas as imagens originais,
    e não as pré-processadas.

    :param dataset_dir: Um diretório onde o dataset será armazenado.
    :return: None
    """
    if not dataset_dir.exists():
        print("Baixando dataset 2 de imagens e criando diretório...")
        dataset_dir.mkdir()

        kaggle.api.dataset_download_files('azaemon/preprocessed-ct-scans-for-covid19', path=dataset_dir, unzip=True)
        shutil.rmtree(dataset_dir / "Preprocessed CT scans")

        positivo_dir = dataset_dir / "Original CT Scans/pCT"
        negativo_dir = dataset_dir / "Original CT Scans/nCT"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")

        if include_ni:
            non_informative_dir = dataset_dir / "Original CT Scans/NiCT"
            formatar_diretorio(non_informative_dir, dataset_dir / "negativo")

        shutil.rmtree(dataset_dir / "Original CT Scans")
        print("Pronto!\n")
    else:
        print("Diretório de imagens para o dataset 2 já está presente na máquina. Prosseguindo...\n")


def combinar_datasets(i: int = 3,
                      dataset1_train: Path = Path("treinamento1"),
                      dataset2_train: Path = Path("treinamento2"),
                      dataset1_test: Path = Path("teste1"),
                      dataset2_test: Path = Path("teste2"),
                      ) -> None:
    """
    Cria os diretórios de treinamento e teste para o 'Dataset 3' (treinamento3 e teste3)
    combinando as imagens dos datasets 1 e 2.

    As imagens são COPIADAS, garantindo que os datasets 1 e 2 originais permaneçam intactos.

    :param i: O número do novo dataset (padrão é 3).
    :param dataset1_train: Diretório de treinamento do primeiro dataset (e.g., "treinamento1").
    :param dataset2_train: Diretório de treinamento do segundo dataset (e.g., "treinamento2").
    :param dataset2_test:  Diretório de teste do primeiro dataset (e.g., "teste2").
    :param dataset1_test: Diretório de teste do segundo dataset (e.g., "teste2").
    :return: None
    """

    def _copy_with_rename(src_file: Path, dest_dir: Path, is_dataset2: bool) -> None:
        """
        Copia um arquivo para o diretório de destino, renomeando se já existir
        e o arquivo for do Dataset 2.
        """
        dest_path = dest_dir / src_file.name

        if is_dataset2:
            # Lógica de renomeação para arquivos do Dataset 2 que colidem
            nome_base = src_file.stem
            extensao = src_file.suffix
            contador = 1

            while dest_path.exists():
                nome_final = f"{nome_base}_d2_{contador}{extensao}"
                dest_path = dest_dir / nome_final
                contador += 1

        shutil.copy2(src_file, dest_path)

    treino_dir = Path(f"treinamento{i}")
    teste_dir = Path(f"teste{i}")

    diretorios_combinar = [
        (dataset1_train, dataset2_train, treino_dir),
        (dataset1_test, dataset2_test, teste_dir)
    ]

    if treino_dir.exists() and teste_dir.exists():
        print(f"Diretórios de treinamento e teste {i} (combinado) já estão presentes. Prosseguindo...\n")
        return

    print(f"Criando diretórios para treinamento e teste {i} (combinado dos Datasets 1 e 2)...")

    # Utiliza ThreadPoolExecutor para paralelizar a cópia de arquivos
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        for origem1, origem2, destino in diretorios_combinar:
            print(f"\tPreparando diretório {destino}...")

            # --- POSITIVOS ---
            destino_positivo = destino / "positivo"
            destino_positivo.mkdir(parents=True, exist_ok=True)

            # Submissão de cópias do Dataset 1 (is_dataset2=False)
            for arquivo in (origem1 / "positivo").iterdir():
                if arquivo.is_file():
                    futures.append(executor.submit(_copy_with_rename, arquivo, destino_positivo, False))

            # Submissão de cópias do Dataset 2 (is_dataset2=True)
            for arquivo in (origem2 / "positivo").iterdir():
                if arquivo.is_file():
                    futures.append(executor.submit(_copy_with_rename, arquivo, destino_positivo, True))

            # --- NEGATIVOS ---
            destino_negativo = destino / "negativo"
            destino_negativo.mkdir(parents=True, exist_ok=True)

            # Submissão de cópias do Dataset 1 (is_dataset2=False)
            for arquivo in (origem1 / "negativo").iterdir():
                if arquivo.is_file():
                    futures.append(executor.submit(_copy_with_rename, arquivo, destino_negativo, False))

            # Submissão de cópias do Dataset 2 (is_dataset2=True)
            for arquivo in (origem2 / "negativo").iterdir():
                if arquivo.is_file():
                    futures.append(executor.submit(_copy_with_rename, arquivo, destino_negativo, True))

        # Espera que todas as cópias sejam concluídas
        for future in futures:
            # Chama .result() para capturar qualquer exceção que tenha ocorrido durante a cópia
            future.result()

    print(f"Combinação de datasets (Dataset {i}) concluída!\n")


def mover_imagens(origem: Path, num_imagens: int, destino: str) -> None:
    """
    Move um número especificado de imagens de um diretório de origem para um destino.

    :param origem: Diretório de origem contendo as imagens.
    :param num_imagens: O número de imagens para mover.
    :param destino: Nome do diretório de destino (incluindo subdiretório para a classe).
    :return: None
    """
    destino_path = Path(destino)
    destino_path.mkdir(parents=True, exist_ok=True)

    for count, image_name in enumerate(origem.iterdir()):
        if count >= num_imagens:
            break
        shutil.move(origem / image_name.name, destino_path / image_name.name)


def treinamento_e_teste(i: int, num_positivas: int, num_negativas: int, test_split: float, dataset_dir: Path) -> None:
    """
    Verifica se os diretórios de treinamento e teste já existem. Se não existirem, cria um diretório temporário,
    copia as imagens do dataset original para esse diretório e então distribui as imagens entre diretórios
    de treinamento e teste conforme a proporção "test_split".

    :param i: Número do dataset, utilizado para nomear os diretórios de treinamento e teste.
    :param num_positivas: Número de exemplos positivos no dataset original.
    :param num_negativas: Número de exemplos negativos no dataset original.
    :param test_split: Proporção dos dados a serem utilizados para teste.
    :param dataset_dir: Caminho do diretório contendo o dataset original.
    :return: None
    """
    treino_dir = Path(f"treinamento{i}")
    teste_dir = Path(f"teste{i}")

    if not (treino_dir.exists() and teste_dir.exists()):
        print(f"Criando diretórios para treinamento e teste {i}...")

        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)

        if not (temp_dir / "positivo").exists():
            print("\tCriando diretório temporário...")
            shutil.copytree(dataset_dir / "positivo", temp_dir / "positivo")
            shutil.copytree(dataset_dir / "negativo", temp_dir / "negativo")
            print("\tPronto!\nProsseguindo...")

        mover_imagens(
            temp_dir / "positivo",
            int(num_positivas * test_split),
            (teste_dir / "positivo").__str__()
        )
        mover_imagens(
            temp_dir / "negativo",
            int(num_negativas * test_split),
            (teste_dir / "negativo").__str__()
        )
        mover_imagens(
            temp_dir / "positivo",
            num_positivas - int(num_positivas * test_split),
            (treino_dir / "positivo").__str__()
        )
        mover_imagens(
            temp_dir / "negativo",
            num_negativas - int(num_negativas * test_split),
            (treino_dir / "negativo").__str__()
        )

        shutil.rmtree(temp_dir)
        print("Pronto!\n")
    else:
        print(f"Diretórios de treinamento e teste {i} já estão presentes. Prosseguindo...\n")


def apagar_treinamento_e_teste():
    pai = Path(".")
    deletar = []

    for diretorio in pai.glob("*"):
        if diretorio.is_dir() and diretorio.name.startswith("teste") or diretorio.name.startswith("treinamento"):
            deletar.append(diretorio)

    for diretorio in deletar:
        shutil.rmtree(diretorio)


def salvar_amostras(dataset, dataset_num, class_names=None, n=9, out_dir="./"):
    """
    Salva grids 3x3 de imagens amostradas de um dicionário.
    Inferimos automaticamente se o dataset é bruto (image/label) ou processado (pixel_values/labels).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Decide tipo com base nas colunas
    colunas = dataset.column_names
    if "image" in colunas:
        tipo = "raw"
    elif "pixel_values" in colunas:
        tipo = "processado"
    else:
        raise ValueError(
            f"Dataset {dataset_num} não contém colunas esperadas. Keys: {colunas}; Esperado: \"image\" ou \"pixel_values\"")
    print(
        f"Salvando amostras do Dataset {dataset_num} {'pré-processado' if tipo == 'processado' else ''}...")

    # Seleciona índices aleatórios
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    imagens, labels = [], []
    for i in indices:
        exemplo = dataset[i]

        if tipo == "raw":
            img = exemplo["image"]
            label = exemplo["label"]
            if hasattr(dataset.features["label"], "int2str"):
                label = dataset.features["label"].int2str(label)
            else:
                label = str(label)

        elif tipo == "processado":
            arr = exemplo["pixel_values"].numpy() if hasattr(exemplo["pixel_values"], "numpy") else exemplo[
                "pixel_values"]
            arr = ((arr.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype("uint8")
            img = arr
            label = class_names[exemplo["labels"]] if class_names else str(exemplo["labels"])

        imagens.append(img)
        labels.append(label)

    # Plotar
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle(f"Amostras{' pŕe-processadas' if tipo == 'processado' else ''} do dataset {dataset_num}",
                 fontsize=14)
    for ax, img, label in zip(axes.flatten(), imagens, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    for ax in axes.flatten()[len(imagens):]:
        ax.axis("off")
    plt.tight_layout()

    # Nome do arquivo reflete automaticamente o tipo
    out_path = os.path.join(out_dir, f"amostras-{tipo}_{dataset_num}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    N = preparar_diretorios(test_split=0.1)
    for i, pair in enumerate(N):
        positivas = pair[0]
        negativas = pair[1]
        total = positivas + negativas
        print(
            f"Dataset {i}: {positivas} imagens positivas, {negativas} imagens negativas ({positivas / total * 100:.2f}%)")
