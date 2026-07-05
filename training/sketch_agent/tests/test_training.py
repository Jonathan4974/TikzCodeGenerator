import importlib


def test_sketch_agent_package_exposes_training_entrypoints():
    module = importlib.import_module("training.sketch_agent")

    assert hasattr(module, "build_training_config")
    assert hasattr(module, "SketchAgentDataset")
    assert hasattr(module, "SketchAgentModelLoader")
    assert hasattr(module, "SketchAgentTrainer")
