import yaml
import argparse
from tester import Tester
from utils.subset_sampling import SUBSET_STRATEGIES
from utils.evaluation_paths import resolve_evaluation_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run name or directory; config.yaml is resolved from this run.",
    )
    parser.add_argument(
        "--active_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path/name, or current/best/best_val when --run is given.",
    )
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
        help="Evaluate on a subset of the test set with this many samples.",
    )
    parser.add_argument(
        "--test_subset_seed",
        type=int,
        default=None,
        help="Random seed used when --test_subset_size is set.",
    )
    parser.add_argument(
        "--test_subset_strategy",
        choices=SUBSET_STRATEGIES,
        default=None,
        help="Subset selection strategy; use style_stratified for quick letter evaluation.",
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

    resolved = resolve_evaluation_paths(
        run=known_args.run,
        active_checkpoint=known_args.active_checkpoint,
        config=known_args.config,
    )

    # Load config file
    with open(resolved["config"], "r") as f:
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
    if known_args.test_subset_strategy is not None:
        config["test_subset_strategy"] = known_args.test_subset_strategy
    if resolved["active_checkpoint"] is not None:
        config["active_checkpoint"] = resolved["active_checkpoint"]
    elif not config.get("active_checkpoint"):
        parser.error(
            "No checkpoint resolved. Provide --active_checkpoint or use --run "
            "with current_checkpoint.json."
        )

    print("Resolved config:", resolved["config"])
    print("Resolved checkpoint:", config["active_checkpoint"])

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
