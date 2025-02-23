from loguru import logger

from xares.task import TaskConfig


def librispeech_asr_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        evalset_size=2620,
        batch_size_train=1,
        disabled=True,
        do_knn=False,
        formal_name="LibriSpeech-100h",
        gradient_accumulation_steps=32,
        label_processor=None,
        metric="WER",
        name="librispeech",
        pretrained_dependencies=["qwen2"],
        task_type="asr",
        test_split="test-clean",
        train_split="train-clean-100",
        use_mini_dataset=False,
        valid_split="dev-clean",
        zenodo_id="14716252",
    )

    if config.use_mini_dataset:
        logger.warning(f"Dataset {config.name} uses mini version for faster evaluation.")
        config.audio_tar_name_of_split[config.train_split] = "train-clean-100-000000.tar"

    return config
