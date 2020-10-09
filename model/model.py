from datasets.Datasets import SNLI_dataset
import pytorch_lightning as pl
from build.config import num_classes,enum_definations
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from typing import Union, List
from torch.utils.data import DataLoader, Dataset



class BertModel(pl.LightningModule):
    def __init__(self, model_path = None, num_classes = num_classes, batch_size = 8, learning_rate = 2e-5, data_base_path = "./data/snli_1.0"):
        super().__init__()
        if(model_path is None):
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels= num_classes)
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.batch_size = batch_size
        self.lr = learning_rate
        self.data_base_path = data_base_path
    def forward(self, X):
        input_ids = X["input_ids"]
        attention_mask = X["attention_mask"]
        token_type_ids = X["token_type_ids"]

        result = self.model(input_ids, token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )

        return result
        

    def _common_step(self,batch, batch_idx):
        inputs , labels = batch
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        loss, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        return loss


    
    def training_step(self, batch : dict, batch_idx):
        loss = self._common_step(batch = batch, batch_idx= batch_idx)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,batch, batch_idx):
        loss = self._common_step(batch = batch, batch_idx= batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    



    



    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr= self.lr,
                )
        return optimizer




    def train_dataloader(self) -> DataLoader:
        snli = SNLI_dataset(self.data_base_path, "train")
        dataloader = DataLoader(dataset= snli , batch_size= self.batch_size,shuffle= True,collate_fn= snli.collate_fn, drop_last= True)
        return dataloader
    

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        snli = SNLI_dataset(self.data_base_path, "dev")
        dataloader = DataLoader(dataset= snli , batch_size= self.batch_size,collate_fn= snli.collate_fn, drop_last= True)
        return dataloader


    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        snli = SNLI_dataset(self.data_base_path, "dev")
        dataloader = DataLoader(dataset= snli , batch_size= self.batch_size,collate_fn= snli.collate_fn, drop_last= True)
        return dataloader
