from dataset.lowercase_letters.split_train_val import (
    split_ood_few_shot,
    split_train_val_test,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/uppercase_letters_")
    parser.add_argument("--save_dir", type=str, default="../data/UppercaseLetters")
    parser.add_argument("--val_percentage", type=float, default=0.1)
    parser.add_argument("--test_percentage", type=float, default=0.1)
    parser.add_argument("--split_for_ood", action="store_true")

    args = parser.parse_args()
    if args.split_for_ood:
        split_ood_few_shot(args.data_dir, args.save_dir, n_shots=[1, 5, 10])
    else:
        split_train_val_test(
            args.data_dir,
            args.save_dir,
            args.val_percentage,
            args.test_percentage,
        )
