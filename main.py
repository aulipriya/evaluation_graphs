from evaluate import create_training_vs_validation_loss_curve, evaluate_models
from utills import load_config_from_yaml

if __name__ == "__main__":
    example_config = load_config_from_yaml("example_data.yaml")
    create_training_vs_validation_loss_curve(
        example_config["training"],
        example_config["validation"],
        "Training Vs Validation Loss for Experiment 1",
        "Epochs",
        "Loss",
        "loss_graph.jpg",
    )
    evaluate_models("config.yaml")
