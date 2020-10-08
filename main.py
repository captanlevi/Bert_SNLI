from pytorch_lightning.trainer import Trainer
from model.model import BertModel
from build.config import trainer_args


if __name__ == "__main__":
    trainer = Trainer(**trainer_args)



