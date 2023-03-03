import dataclasses

import numpy as np
import torch 
from torch import nn
from transformers import (
    PretrainedConfig, 
    PreTrainedModel, 
    TrainingArguments,
    Trainer,
)

# copied from tests/trainer/test_trainer.py and modified
class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42):
        np.random.seed(seed)
        self.label_names = ["labels"]
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,))
                   for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


# copied from tests/trainer/test_trainer.py and modified
class RegressionModelConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.random_torch = random_torch
        self.hidden_size = 1


# copied from tests/trainer/test_trainer.py and modified
class RegressionPreTrainedModel(PreTrainedModel):
    config_class = RegressionModelConfig
    base_model_prefix = "regression"

    def __init__(self, config):
        super().__init__(config)
        self.a = nn.Parameter(torch.tensor(config.a).float())
        self.b = nn.Parameter(torch.tensor(config.b).float())

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y, )
        loss = nn.functional.mse_loss(y, labels)
        return loss, y


# copied from tests/trainer/test_trainer.py
@dataclasses.dataclass
class RegressionTrainingArguments(TrainingArguments):
    a: float = 0.0
    b: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        # save resources not dealing with reporting (also avoids the warning when it's not set)
        self.report_to = []


def main():
    bs = 8
    a = torch.ones(1000, bs) + 0.001
    b = torch.ones(1000, bs) - 0.001

    train_dataset = RegressionDataset(length=64)
    eval_dataset = RegressionDataset(length=16)
    config = RegressionModelConfig(a=a, b=b, double_output=False)
    model = RegressionPreTrainedModel(config)
    args = RegressionTrainingArguments(output_dir="./regression", a=a, b=b,
                                       skip_memory_metrics=False)

    trainer = Trainer(
        model, args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    #trainer.train()                # error
    metrics = trainer.evaluate()    # error


if __name__ == "__main__":
    main()
