import argparse
import json
from pathlib import Path
import random
from tqdm import tqdm
import spacy
import ray

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

random.seed(42)
spacy.cli.download("en_core_web_sm")

ray_excludes = ['/src/data/train_paragraphs_processed_data.json',
                '/src/data/train_entailment_processed_data.json',
                '/src/data/task1_train_files_2024.zip',
                '/src/data/test_paragraphs_processed_data.json',
                '/src/data/task1_test_files_2024.zip',
                '/src/data/task2_train_files_2024.zip',
                '/src/output/model/task2/1.pkl',
                '/src/output/checkpoints/hdbscan.joblib']
def read_text_file(file_path):
    """Read text file content"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_sentences(text, method="spacy"):
    """Extract sentences from text using specified method"""
    if method == "spacy":
        return get_spacy_sentences(text)
    elif method == "sumy":
        return get_sumy_sentences(text)
    else:
        raise ValueError(f"Unknown sentence extraction method: {method}")


def get_spacy_sentences(text):
    """Extract sentences using spaCy"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def get_sumy_sentences(text):
    """Extract sentences using Sumy"""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    return [str(sentence) for sentence in parser.document.sentences]


def generate_negative_pairs(labels, files):
    """Generate negative examples by randomly pairing documents"""
    negative_pairs = []
    all_docs = list(files)

    for q_file in tqdm(labels.keys(), desc="Generating negative pairs"):
        # Get all possible p_files excluding those that are positive pairs
        available_docs = [doc for doc in all_docs if doc not in labels[q_file]]

        # Generate same number of negative pairs as positive pairs
        num_negatives = len(labels[q_file])
        if available_docs:
            negative_pairs.extend(
                [(q_file, random.choice(available_docs)) for _ in range(num_negatives)]
            )

    return negative_pairs


def process_files(files_path, labels_file, output_file, sentence_method="spacy"):
    # Read labels file
    with open(labels_file, "r") as f:
        labels = json.load(f)

    # Get all files in directory
    files = set(Path(files_path).glob("*.txt"))
    files = {f.name for f in files}

    # Generate positive and negative pairs
    positive_pairs = [(q, p) for q in labels for p in labels[q]]
    negative_pairs = generate_negative_pairs(labels, files)

    # Process all pairs
    output_data = []

    # Process positive pairs
    ray.init(num_cpus=2, ignore_reinit_error=True, runtime_env={'excludes': ray_excludes})

    @ray.remote
    def process_pair(
        q_file, p_file, files_path, positive_pairs_set, sentence_method="spacy"
    ):
        q_content = read_text_file(Path(files_path) / q_file)
        p_content = read_text_file(Path(files_path) / p_file)
        q_sentences = get_sentences(q_content, method=sentence_method)
        p_sentences = get_sentences(p_content, method=sentence_method)
        entry = {
            "guid": f"{q_file.split('.')[0]}_{p_file.split('.')[0]}",
            "q_paras": q_sentences,
            "c_paras": p_sentences,
            "label": 1 if (q_file, p_file) in positive_pairs_set else 0,
        }
        return entry

    positive_pairs_set = set(positive_pairs)
    all_pairs = positive_pairs + negative_pairs

    # Ray cannot serialize spaCy models directly, so load it inside the remote function
    # We'll pass nlp=None and reload inside process_pair if needed
    # For efficiency, you can use ray's object store or actor, but for simplicity:
    def process_pair_wrapper(
        q_file, p_file, files_path, positive_pairs_set, sentence_method="spacy"
    ):
        return process_pair.remote(
            q_file,
            p_file,
            files_path,
            positive_pairs_set,
            sentence_method=sentence_method,
        )

    batch_size = 10
    output_data = []
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Batching Ray tasks"):
        batch = all_pairs[i : i + batch_size]
        futures = [
            process_pair_wrapper(
                q_file, p_file, files_path, positive_pairs_set, sentence_method=sentence_method
            )
            for q_file, p_file in tqdm(batch, desc="Dispatching Ray tasks", leave=False)
        ]
        output_data.extend(ray.get(futures))

    # Write output line by line
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Processed {len(output_data)} entries")

def serial_run(files_path, labels_file, output_file, sentence_method="spacy"):
        # Read labels file
    with open(labels_file, "r") as f:
        labels = json.load(f)

    # Get all files in directory
    files = set(Path(files_path).glob("*.txt"))
    files = {f.name for f in files}

    # Generate positive and negative pairs
    positive_pairs = [(q, p) for q in labels for p in labels[q]]
    negative_pairs = generate_negative_pairs(labels, files)
    positive_pairs_set = set(positive_pairs)
    all_pairs = positive_pairs + negative_pairs
    # Process all pairs
    output_data = []
    for q_file, p_file in tqdm(all_pairs, desc="Processing pairs"):
        q_content = read_text_file(Path(files_path) / q_file)
        p_content = read_text_file(Path(files_path) / p_file)
        q_sentences = get_sentences(q_content, method=sentence_method)
        p_sentences = get_sentences(p_content, method=sentence_method)
        entry = {
            "guid": f"{q_file.split('.')[0]}_{p_file.split('.')[0]}",
            "q_paras": q_sentences,
            "c_paras": p_sentences,
            "label": 1 if (q_file, p_file) in positive_pairs_set else 0,
        }
        output_data.append(entry)

    # Write output line by line
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Processed {len(output_data)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/task1_train_files_2024/text_rank_summaries/",
        help="Caminho para o diretório de entrada (padrão: data/task1_train_files_2024/text_rank_summaries/)",
    )

    parser.add_argument(
        "--labels-file",
        type=str,
        default="data/task1_train_labels_2024.json",
        help="Caminho para o arquivo de rótulos (padrão: data/task1_train_labels_2024.json)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="data/train_sumy_paragraphs_processed_data.json",
        help="Caminho para o arquivo de saída (padrão: data/train_sumy_paragraphs_processed_data.json)",
    )

    parser.add_argument(
        "--sentence-method",
        type=str,
        default="spacy",
        choices=["spacy", "sumy"],
        help="Método de extração de sentenças a ser usado (padrão: spacy)",
    )
    args = parser.parse_args()
    files_path = args.input_dir
    labels_file = args.labels_file
    output_file = args.output_file
    sentence_method = args.sentence_method

    serial_run(files_path, labels_file, output_file, sentence_method=sentence_method)
    # process_files(files_path, labels_file, output_file, sentence_method=sentence_method)
