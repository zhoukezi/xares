from xares.task import TaskConfig


def key_scratching_config(encoder) -> TaskConfig:
    class_label_maps = {
        "T": 0,
        "F": 1,
    }

    return TaskConfig(
        encoder=encoder,
        name="key_scratching",
        private=True,
        epochs=50,
        label_processor=lambda x: class_label_maps[x["tag"]],
        output_dim=len(class_label_maps),
        train_split="key_scratching_train",
        test_split="key_scratching_test",
        valid_split="key_scratching_test",
        eval_weight=4154,
    )
