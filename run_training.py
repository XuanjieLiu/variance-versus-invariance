import yaml
import argparse
from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
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
    if known_args.name is not None:
        config["name"] = known_args.name
    if known_args.debug:
        config["debug"] = True

    trainer = Trainer(config)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
