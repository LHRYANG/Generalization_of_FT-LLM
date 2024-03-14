import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
#import utils 

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_MAP = {
        "summary": "### Input:\n{input}\n### Summary:",
        "question": "### Input:\n{input}\n### Answer:\n{answer}\n### Question:",
        "sentiment": "### Input:\n{input}\n### Sentiment:",
        "inference": "### Input_1:\n{input_1}\n### Input_2:\n{input_2}\n### Inference:",
        "detection": "### Input_1:\n{input_1}\n### Input_2:\n{input_2}\n### Paraphrase Classification:"
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, shot=2):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        #prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        if "summary" in data_path:
            prompt_input = PROMPT_MAP["summary"]
            prompt_path = "data/summary/few_shot/xsum_few_shot.json"
        if "question" in data_path:
            prompt_input = PROMPT_MAP["question"]
            prompt_path = "data/question/few_shot/socialqa_few_shot.json"
        if "sentiment" in data_path:
            prompt_input = PROMPT_MAP["sentiment"]
            prompt_path = "data/sentiment/few_shot/amazon_few_shot.json"
        if "inference" in data_path:
            prompt_input = PROMPT_MAP["inference"]
            prompt_path = "data/inference/few_shot/multinli_few_shot.json"
        if "detection" in data_path:
            prompt_input = PROMPT_MAP["detection"]
            prompt_path = "data/detection/few_shot/paws_few_shot.json"

        with open(prompt_path) as f:
            template = json.load(f)

        context = ""
        for tmp in template[:shot]:
            context += prompt_input.format_map(tmp) + tmp["output"] + "\n\n"

        print("context for shot ",str(shot), context)
        sources = [
            context+prompt_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        print("total number of training examples: ",len(sources))
        print("sources: ",sources[0:2])
        print("targets: ",targets[0:2])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        print(self.input_ids[0:2])
        self.labels = data_dict["labels"]
        print(self.labels[0:2])
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args,shot=2) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path,shot=shot)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "summary" in data_args.data_path or "question" in data_args.data_path:
        shots = [2,1]
    elif "detection" in data_args.data_path or "sentiment" in data_args.data_path:
        shots = [2,4]
    else:
        shots = [3,5]

    for shot in shots:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        special_tokens_dict = dict()
        #print(tokenizer.pad_token_id,tokenizer.pad_token,tokenizer.eos_token,tokenizer.eos_token_id,tokenizer.bos_token,tokenizer.bos_token_id,tokenizer.unk_token,tokenizer.unk_token_id)
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        #print(tokenizer.pad_token_id,tokenizer.pad_token,tokenizer.eos_token,tokenizer.eos_token_id,tokenizer.bos_token,tokenizer.bos_token_id,tokenizer.unk_token,tokenizer.unk_token_id)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,shot=shot)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.train()
        #trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir+str(shot))


if __name__ == "__main__":
    train()
