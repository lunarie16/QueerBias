import optuna
from main import LanguageModelTrainer


def objective(trial):
    model_name = trial.suggest_categorical("model_name", ["gpt2", "distilgpt2"])
    mode = trial.suggest_categorical("mode", ["soft-prompt", "fine-tuning"])
    prompt_length = trial.suggest_int("prompt_length", 5, 20)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    batch_size = trial.suggest_int("batch_size", 4, 16)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    dataset_path = "path/to/dataset.csv"  # Update with actual dataset path

    trainer = LanguageModelTrainer(model_name=model_name, mode=mode, prompt_length=prompt_length)
    trainer.train(dataset_path=dataset_path, num_train_epochs=num_train_epochs,
                  batch_size=batch_size, learning_rate=learning_rate)
    eval_results = trainer.evaluate(dataset_path=dataset_path)

    return eval_results["eval_loss"]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
