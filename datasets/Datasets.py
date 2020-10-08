import os
import sys
import json
from torch.utils.data import Dataset, DataLoader
from build.config import enum_definations
from transformers import BertTokenizer
import torch

class SNLI_dataset(Dataset):
    def __init__(self,base_path : str, mode : str):
        """
        base path is to the folder of snli_1.0
        """
        super().__init__()
        assert mode in ["train", "dev", "test"] , "mode can be train test or dev only"
        data_path = os.path.join(base_path , "snli_1.0_" + mode + ".jsonl")
        with open(data_path , "r") as F:
            json_list = list(F)
        
        self.data = []
        label_map = {"neutral" : enum_definations.neutral_label, "entailment" : enum_definations.positive_label, "contradiction" : enum_definations.negative_label}
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        for json_string in json_list:
            dct = json.loads(json_string)
            if(dct["gold_label"] == '-'):
                continue
            new_dct = dict(
                sentence1 = dct["sentence1"],
                sentence2 = dct["sentence2"],
                label = label_map[dct["gold_label"]]

            )
            self.data.append(new_dct)

            


    def __getitem__(self, index: int):
        return self.data[index]


    def __len__(self):
        return len(self.data)

    

    def collate_fn(self,batch):
        sentences1 = [item["sentence1"] for item in batch]
        sentences2 = [item["sentence2"] for item in batch]
        labels = [item["label"] for item in batch]


        batch_inputs = self.tokenizer(sentences1, sentences2, padding= "longest", return_tensors= "pt")
        batch_outputs = torch.tensor(labels, dtype= torch.long)

        return batch_inputs, batch_outputs




    