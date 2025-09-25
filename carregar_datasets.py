import shutil
from pathlib import Path

import kaggle
from datasets import load_dataset, DatasetDict
from transformers import BaseImageProcessor


def carregar_datasets(i: int, validation_split: float, processor: BaseImageProcessor) -> \
        tuple[
            DatasetDict, list[str]]:
    """
    Carrega e prepara datasets de treinamento, validação e teste para HuggingFace Trainer.
    """
    # Carrega conjuntos de imagens
    raw_datasets = DatasetDict({
        'train': load_dataset("imagefolder", data_dir=f"./treinamento{i}")["train"],
        'test': load_dataset("imagefolder", data_dir=f"./teste{i}")["train"]
    })

    # Split em validação
    split = raw_datasets["train"].train_test_split(test_size=validation_split, seed=42)
    raw_datasets["train"] = split["train"]
    raw_datasets["validation"] = split["test"]

    class_names = raw_datasets["train"].features["label"].names

    def transform(examples):
        if "image" in examples:
            images = [img.convert("RGB") for img in examples["image"]]
            inputs = processor(images)
            inputs["labels"] = examples["label"]
            return inputs
        else:
            raise ValueError(f"Coluna 'image' não está presente. Keys recebidas: {examples.keys()}")

    datasets = raw_datasets.map(transform, batched=True, num_proc=8, remove_columns=["image", "label"])
    return datasets, class_names


def preparar_diretorios(test_split: float, dataset_dirs=None) -> list[list[int]]:
    """
    Prepara os diretórios e conjuntos de dados para treinamento e teste.

    Verifica se os diretórios de datasets específicos existem. Se não, prepara os datasets necessários e
    cria os diretórios de treinamento e teste dividindo as imagens conforme a proporção especificada para teste.

    :param test_split: Proporção dos dados a serem utilizados para teste.
    :param dataset_dirs: Lista dos caminhos para os diretórios de imagens.
    :return: Uma lista contendo sublistas com o número de exemplos positivos e negativos para cada dataset.
    """
    if dataset_dirs is None:
        dataset_dirs = [Path("./dataset1"), Path("./dataset2")]
    N = []

    for i, dataset_dir in enumerate(dataset_dirs, start=1):
        if i == 1:
            preparar_dataset1(dataset_dir)
        elif i == 2:
            preparar_dataset2(dataset_dir)

        num_positivas = len(list((dataset_dir / "positivo").glob("*")))
        num_negativas = len(list((dataset_dir / "negativo").glob("*")))
        treinamento_e_teste(i, num_positivas, num_negativas, test_split, dataset_dir)
        N.append([num_positivas, num_negativas])

    return N


def formatar_diretorio(origem: Path, destino: Path) -> None:
    """
    Move todos os arquivos de um diretório de origem para um destino e remove o diretório de origem.

    :param origem: Diretório de origem contendo os arquivos.
    :param destino: Diretório de destino.
    :return: None
    """
    destino.mkdir(parents=True, exist_ok=True)
    for file in origem.iterdir():
        shutil.move(file, destino / file.name)
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


def preparar_dataset2(dataset_dir: Path = Path("./dataset2")) -> None:
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
        non_informative_dir = dataset_dir / "Original CT Scans/NiCT"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")
        formatar_diretorio(non_informative_dir, dataset_dir / "negativo")

        shutil.rmtree(dataset_dir / "Original CT Scans")
        print("Pronto!\n")
    else:
        print("Diretório de imagens para o dataset 2 já está presente na máquina. Prosseguindo...\n")


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

        mover_imagens(temp_dir / "positivo", int(num_positivas * test_split), (teste_dir / "positivo").__str__())
        mover_imagens(temp_dir / "negativo", int(num_negativas * test_split), (teste_dir / "negativo").__str__())
        mover_imagens(temp_dir / "positivo", num_positivas - int(num_positivas * test_split),
                      (treino_dir / "positivo").__str__())
        mover_imagens(temp_dir / "negativo", num_negativas - int(num_negativas * test_split),
                      (treino_dir / "negativo").__str__())

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


if __name__ == "__main__":
    preparar_diretorios(test_split=0.1)