# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import os
import argparse

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
from tqdm import tqdm

nltk.download("punkt_tab")

LANGUAGE = "english"


def main(dirpath, output_path, ratio):
    os.makedirs(output_path, exist_ok=True)

    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    for f in tqdm(files, desc="Summarizing files"):
        file_path = os.path.join(dirpath, f)
        parser = PlaintextParser.from_file(file_path, Tokenizer(LANGUAGE))
        document = parser.document
        total_sentences = len(document.sentences)
        sentences_count = max(1, min(total_sentences, int(total_sentences * ratio)))
        summary = ""
        for sentence in summarizer(document, sentences_count):
            summary += f" {str(sentence)}"

        with open(output_path + f, "w", encoding="utf-8") as f:
            f.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/task1_train_files_2024/task1_train_files_2024/",
        help="Caminho para o diretório de entrada (padrão: data/task1_train_files_2024/task1_train_files_2024/)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/task1_train_files_2024/text_rank_summaries/",
        help="Caminho para o diretório de saída (padrão: data/task1_train_files_2024/text_rank_summaries/)",
    )

    parser.add_argument(
        "--sumy-ratio",
        type=float,
        default=0.5,
        help="Proporção de resumo para o Sumy (padrão: 0.5)",
    )

    args = parser.parse_args()
    dirpath = args.input_dir
    output_path = args.output_dir
    sumy_ratio = args.sumy_ratio
    main(dirpath, output_path, sumy_ratio)
