import argparse
import string

from dataset.lowercase_letters.letter_office import LetterOffice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate uppercase letter images")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/uppercase_letters_",
        help="Directory to save generated uppercase letter images",
    )
    parser.add_argument(
        "--n_pages",
        type=int,
        default=26000,
        help="Number of pages to generate",
    )
    args = parser.parse_args()

    press = LetterOffice(
        pagesize="26x1",
        patchsize="32x48",
        output_dir=args.save_dir,
        alphabet=string.ascii_uppercase,
        dataset_name="uppercase letter",
    )
    press.generate_folder(n_pages=args.n_pages)
    print(f"Generated {args.n_pages} pages of uppercase letter images")
    print(f"Saved to {args.save_dir}")
