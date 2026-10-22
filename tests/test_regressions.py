"""Regression checks for frozen features, loss-based stopping, and model selection.

Run with: python -m unittest discover -s tests -v
Uses synthetic data. App integration checks require existing local checkpoints
and are skipped when none are available, so tests do not download model weights.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import train
from models import create_model, freeze_backbone, unfreeze_model, unfreeze_last_block, set_training_mode


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1))
        self.layer4 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.fc = nn.Linear(4, 2)

    def forward(self, images):
        return self.fc(self.layer4(self.features(images)).flatten(1))


def notebook_training_namespace():
    notebook = json.loads((ROOT / "notebooks/02_training.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and index != 14:
            exec(compile("".join(cell["source"]), f"notebook-cell-{index}", "exec"), namespace)
    return namespace


class TrainingTests(unittest.TestCase):
    def test_partial_finetuning_updates_last_block_only_in_both_implementations(self):
        notebook = notebook_training_namespace()
        for implementation in ("src", "notebook"):
            with self.subTest(implementation=implementation):
                model = TinyModel()
                (unfreeze_last_block if implementation == "src" else notebook["unfreeze"])(model)
                before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=.01)
                batches = [(torch.rand(8, 3, 8, 8), torch.arange(8) % 2)]
                args = (model, batches, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"),
                        torch.amp.GradScaler("cpu", enabled=False))
                if implementation == "src":
                    train.train_one_epoch(*args)
                else:
                    notebook["run_epoch"](*args, train=True)
                for name, tensor in model.features.state_dict().items():
                    self.assertTrue(torch.equal(before["features." + name], tensor), name)
                self.assertFalse(model.features.training)
                self.assertTrue(model.layer4.training)
                self.assertFalse(torch.equal(before["layer4.0.weight"], model.layer4[0].weight))
                self.assertFalse(torch.equal(before["layer4.1.running_mean"], model.layer4[1].running_mean))
                self.assertFalse(torch.equal(before["fc.weight"], model.fc.weight))

    def test_actual_architectures_freeze_early_stages_in_both_implementations(self):
        notebook = notebook_training_namespace()
        for name in ("resnet18", "resnet50", "efficientnet_b0"):
            for implementation in ("src", "notebook"):
                with self.subTest(model=name, implementation=implementation):
                    model = create_model(name, pretrained=False)
                    (unfreeze_last_block if implementation == "src" else notebook["unfreeze"])(model)
                    for key, parameter in model.named_parameters():
                        expected = key.startswith(("layer4.", "fc.")) if name.startswith("resnet") else key.startswith(("features.7.", "features.8.", "classifier."))
                        self.assertEqual(parameter.requires_grad, expected, key)
                    set_training_mode(model)
                    frozen = model.layer1 if name.startswith("resnet") else model.features[0]
                    active = model.layer4 if name.startswith("resnet") else model.features[7]
                    self.assertFalse(frozen.training)
                    self.assertTrue(active.training)

    def test_evaluation_notebook_keeps_results_in_memory(self):
        notebook = json.loads((ROOT / "notebooks/03_evaluation.ipynb").read_text(encoding="utf-8"))
        namespace = {}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code" and index != 8:
                exec(compile("".join(cell["source"]), f"evaluation-cell-{index}", "exec"), namespace)
        with tempfile.TemporaryDirectory(dir=ROOT / "models") as directory:
            output = Path(directory)
            model = TinyModel()
            model.fc = nn.Linear(4, 200)
            torch.save({"model_state_dict": model.state_dict(), "validation_accuracy": .7}, output / "resnet50_best.pth")
            dataset = torch.utils.data.TensorDataset(torch.rand(200, 3, 8, 8), torch.arange(200))
            loader = torch.utils.data.DataLoader(dataset, batch_size=32)
            namespace.update(CK=output, test_loader=lambda *args: loader, model_of=lambda name: model)
            notebook["cells"][8]["source"] = ["".join(notebook["cells"][8]["source"]).replace("MODEL = 'efficientnet_b0'", "MODEL = 'resnet50'")]
            with patch.object(namespace["plt"], "show"), contextlib.redirect_stdout(io.StringIO()):
                exec("".join(notebook["cells"][8]["source"]), namespace)
            namespace["plt"].close("all")
            self.assertEqual(namespace["results"]["classes_evaluated"], 200)
            self.assertEqual(namespace["results"]["test_samples"], 200)
            self.assertEqual(list(output.rglob("*.json")), [])

    def test_frozen_batchnorm_and_classifier_updates_in_both_implementations(self):
        notebook = notebook_training_namespace()
        for implementation in ("src", "notebook"):
            with self.subTest(implementation=implementation):
                model = TinyModel()
                freeze_backbone(model)
                before_bn = model.features[1].running_mean.clone()
                before_head = model.fc.weight.detach().clone()
                batches = [(torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 0, 1]))]
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
                scaler = torch.amp.GradScaler("cpu", enabled=False)
                args = (model, batches, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), scaler)
                if implementation == "src":
                    train.train_one_epoch(*args)
                else:
                    notebook["run_epoch"](*args, train=True)
                self.assertTrue(torch.equal(before_bn, model.features[1].running_mean))
                self.assertFalse(torch.equal(before_head, model.fc.weight))
                unfreeze_model(model)
                if implementation == "src":
                    train.train_one_epoch(*args)
                else:
                    notebook["run_epoch"](*args, train=True)
                self.assertFalse(torch.equal(before_bn, model.features[1].running_mean))

    def test_accuracy_improvements_do_not_reset_loss_patience(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "models") as directory:
            output = Path(directory)
            validations = [(3., .1), (2., .2), (2.1, .3), (2.2, .4)]
            with contextlib.ExitStack() as stack:
                for name, value in {
                    "WARMUP_EPOCHS": 1, "MAX_EPOCHS": 10, "EARLY_STOPPING_PATIENCE": 2,
                    "CHECKPOINT_DIR": output / "checkpoints", "HISTORY_DIR": output / "history",
                }.items():
                    stack.enter_context(patch.object(train, name, value))
                stack.enter_context(patch.object(train, "create_dataloaders", return_value=(SimpleNamespace(dataset=[1]), SimpleNamespace(dataset=[1]), [])))
                stack.enter_context(patch.object(train, "create_model", return_value=TinyModel()))
                stack.enter_context(patch.object(train, "get_device", return_value=torch.device("cpu")))
                stack.enter_context(patch.object(train, "train_one_epoch", return_value=(1., .9)))
                validation = stack.enter_context(patch.object(train, "validate", side_effect=validations))
                with contextlib.redirect_stdout(io.StringIO()):
                    train.train_model("resnet18")
                self.assertEqual(validation.call_count, 4)
            checkpoint = torch.load(output / "checkpoints/resnet18_best.pth", weights_only=True)
            self.assertEqual(checkpoint["validation_accuracy"], .4)
            self.assertEqual(checkpoint["epoch"], 4)
            self.assertEqual([group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]], [1e-5, 1e-4])
            history = json.loads((output / "history/resnet18_history.json").read_text())
            self.assertEqual(len(history["train_loss"]), 4)

            notebook = notebook_training_namespace()
            notebook_model = TinyModel()
            notebook.update(WARMUP=1, MAX_EPOCHS=10, PAT=2, CK=output / "notebook_checkpoints",
                            dev=lambda: torch.device("cpu"), loaders=lambda *args: (SimpleNamespace(dataset=SimpleNamespace(indices=[0])), [1], []),
                            model_of=lambda *args: notebook_model, plot_hist=lambda h: None,
                            CUB=lambda *args: [1], DataLoader=lambda *args, **kwargs: [1])
            values = iter(validations)
            notebook["run_epoch"] = lambda *args: (1., .9) if args[-1] or args[3] is None else next(values)
            before = set(output.rglob("*.json"))
            with contextlib.redirect_stdout(io.StringIO()):
                history = notebook["train_model"]("resnet18")
            self.assertEqual(len(history["tr_loss"]), 4)
            saved_history = output / "notebook_checkpoints/resnet18_history.json"
            self.assertEqual(set(output.rglob("*.json")), before | {saved_history})
            self.assertEqual(json.loads(saved_history.read_text())["va_acc"], [.1, .2, .3, .4])
            checkpoint = torch.load(output / "notebook_checkpoints/resnet18_best.pth", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 4)
            self.assertEqual(checkpoint["validation_accuracy"], .4)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import also checks strict loading of the actual best local checkpoint.
        checkpoint_dir = ROOT / "models/checkpoints"
        if not any(p.suffix.lower() in (".pth", ".pt", ".bin") for p in checkpoint_dir.glob("*")):
            raise unittest.SkipTest("Train a local model before running app integration checks.")
        import app
        cls.app = app

    def test_selection_uses_checkpoint_accuracy_and_changes_with_ranking(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "models") as directory:
            paths = [Path(directory) / f"{name}_best.pth" for name in self.app.SUPPORTED_MODELS]
            for path, name, accuracy in zip(paths, self.app.SUPPORTED_MODELS, [.7, .8, .75]):
                torch.save({"model_name": name, "validation_accuracy": accuracy}, path)
            self.assertEqual(self.app.select_best_checkpoint(paths), str(paths[1]))
            torch.save({"model_name": "efficientnet_b0", "validation_accuracy": .9}, paths[2])
            self.assertEqual(self.app.select_best_checkpoint(paths), str(paths[2]))
            with patch.object(self.app, "LOCAL_CHECKPOINT_DIR", Path(directory)), patch.object(self.app, "HfApi") as api:
                self.assertEqual(self.app.get_checkpoint_path(), str(paths[2]))
                api.assert_not_called()
            with patch.object(self.app, "LOCAL_CHECKPOINT_DIR", Path(directory) / "absent"), \
                 patch.object(self.app, "HfApi") as api, \
                 patch.object(self.app, "hf_hub_download", side_effect=lambda **kw: str(Path(directory) / kw["filename"])):
                api.return_value.list_repo_files.return_value = [p.name for p in paths] + ["backup/old.pth"]
                self.assertEqual(self.app.get_checkpoint_path(), str(paths[2]))

    def test_missing_validation_accuracy_is_not_silently_ranked(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "models") as directory:
            path = Path(directory) / "model.pth"
            torch.save({"model_name": "resnet18"}, path)
            with self.assertRaisesRegex(ValueError, "validation_accuracy"):
                self.app.select_best_checkpoint([path])


if __name__ == "__main__":
    unittest.main()
