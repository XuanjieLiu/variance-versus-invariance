import yaml
import argparse
from tester import Tester


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pr_metrics", action="store_true")
    parser.add_argument("--vis_tsne", action="store_true")
    parser.add_argument("--confusion_mtx", action="store_true")
    parser.add_argument("--zero_shot_ood", action="store_true")
    parser.add_argument("--few_shot_ood", action="store_true")
    parser.add_argument(
        "--test_subset_size",
        type=int,
        default=None,
        help="Evaluate on a random subset of the test set with this many samples.",
    )
    parser.add_argument(
        "--test_subset_seed",
        type=int,
        default=None,
        help="Random seed used when --test_subset_size is set.",
    )
    parser.add_argument(
        "--help",
        action="help",
        help="You can pass any argument from the config file as a command line argument. For example, --optimizer_config.lr 1.0e-3 will set the learning rate to 1.0e-3.",
    )

    # Parse known and unknown arguments
    known_args, unknown_args = parser.parse_known_args()

    # Process unknown arguments as key-value pairs
    additional_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            value = unknown_args[unknown_args.index(arg) + 1]
            additional_args[key] = value

    # Load config file
    with open(known_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update config with additional arguments
    for key, value in additional_args.items():
        value = yaml.safe_load(value)
        if "." in key:
            key1, key2 = key.split(".")
            config[key1][key2] = value
            print(f"Set {key1}.{key2} to {value}")
        else:
            config[key] = value
            print(f"Set {key} to {value}")

    # Set known arguments
    if known_args.debug:
        config["debug"] = True
    if known_args.test_subset_size is not None:
        config["test_subset_size"] = known_args.test_subset_size
    if known_args.test_subset_seed is not None:
        config["test_subset_seed"] = known_args.test_subset_seed

    tester = Tester(config)
    tester.prepare_data()
    tester.build_model()
    tester.test(
        known_args.pr_metrics,
        known_args.vis_tsne,
        known_args.confusion_mtx,
        known_args.zero_shot_ood,
        known_args.few_shot_ood,
    )
