from random import randint

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog

from nuclei_graph.data.data_module import DataModule
from nuclei_graph.nuclei_graph_model import NucleiGraphTransformer


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger) -> None:
    seed_everything(config.seed, workers=True)
    # torch.set_float32_matmul_precision("medium")

    data = instantiate(config.data, _recursive_=False, _target_=DataModule)
    model = instantiate(config.model, _target_=NucleiGraphTransformer)
    trainer = instantiate(config.trainer, _target_=Trainer, logger=logger)
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
