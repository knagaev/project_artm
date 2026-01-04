import argparse
import json

from artm_lib.io.parquet_utils import convert_csv_dir_to_parquet
from artm_lib.preprocessing.tokenizer import simple_tokenizer
from artm_lib.preprocessing.vocabulary import build_vocab_and_index_from_parquet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=False, default="artm_lib/data/csvs")
    parser.add_argument("--parquet_dir", required=False, default="artm_lib/data/parquets")
    parser.add_argument("--vocab_out", default="vocab.json")
    args = parser.parse_args()

    convert_csv_dir_to_parquet(args.csv_dir, args.parquet_dir)
    token_to_id, doc_index = build_vocab_and_index_from_parquet(
        args.parquet_dir, tokenizer=simple_tokenizer
    )
    with open(args.vocab_out, "w") as f:
        json.dump(token_to_id, f)


if __name__ == "__main__":
    main()
