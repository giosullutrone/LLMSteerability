from typing import Dict, Union, List, Callable, Tuple
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset, DownloadMode, load_from_disk, concatenate_datasets
from recipes.DatasetRecipe import DatasetRecipe
import numpy as np
from tqdm import tqdm
from utils import disable_progress_bar_wrapper
from functools import partial


class DatasetDispatcher:
    def __init__(self, dataset_recipe: DatasetRecipe) -> None:
        self._dataset_recipe = dataset_recipe

    def _load_dataset(self, 
                     dataset_path: str,
                     split: str) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        try: dataset = load_from_disk(dataset_path, **self.dataset_recipe.dataset_load)
        except: dataset = load_dataset(dataset_path, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, **self.dataset_recipe.dataset_load)

        # Split dataset if requested
        if split is not None: dataset = dataset[split]
        if self.dataset_recipe.preprocess_dataset: dataset = self.dataset_recipe.preprocess_dataset(dataset)
        return dataset

    def _process_dataset(
            self,
            dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
            dataset_support: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
            postprocess_function,
            eos_token: str,
            include_labels_inside_text: bool,
            dynamic_examples: bool=False,
            num_examples: int=0, 
            shuffle: bool=False, 
            shuffle_seed: int=42, 
            num_proc: Union[int, None]=None
        ) -> Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None],
                   Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]]:

        def create_text(sample: Dict, labels: bool) -> Dict: 
            if labels: sample["text"] = sample["prompts"] + sample["labels"] + eos_token
            else: sample["text"] = sample["prompts"]
            return sample
        
        create_dataset_support = partial(self._process_dataset, 
                                         dataset_support=None, 
                                         postprocess_function=None, 
                                         eos_token="", 
                                         include_labels_inside_text=True, 
                                         num_proc=None)

        @disable_progress_bar_wrapper
        def dynamic_examples_wrapper(samples: Dict):
            num_samples = len(samples[list(samples.keys())[0]])
            results = {}
            for i in range(num_samples):
                _, examples = create_dataset_support(dataset=Dataset.from_dict({k: [x for idx, x in enumerate(samples[k]) if idx!=i] for k in samples}))
                sample = {k: samples[k][i] for k in samples}
                result = self.dataset_recipe.preprocess_function(sample, examples=examples)
                results = {k: [result[k], *results.get(k, [])] for k in result}
            return results
        
        # Shuffle the dataset if requested
        if shuffle: dataset.shuffle(shuffle_seed)

        # Create the dataset support if requested
        if num_examples > 0 and dataset_support is None and not dynamic_examples:
            _, dataset_support = create_dataset_support(dataset.select(range(num_examples)))
            dataset = dataset.select(range(num_examples, dataset.num_rows))

        # Apply the preprocess
        if not dynamic_examples or num_examples == 0:
            # If the dataset support if fixed, we can just provide it to the preprocess function
            dataset = dataset.map(self.dataset_recipe.preprocess_function, remove_columns=dataset.column_names, fn_kwargs={"examples": dataset_support}, num_proc=num_proc, desc="Preprocessing prompts")
        else:
            # If the dataset support is dynamic, we have to call the dynamic wrapper for the example definition
            # Batch the process with batch_size equal to num_examples+1 requested and use the last value as main prompt and the others as examples
            dataset = dataset.map(dynamic_examples_wrapper, remove_columns=dataset.column_names, num_proc=num_proc, batched=True, batch_size=num_examples+1, drop_last_batch=True, desc="Preprocessing prompts")

        # Filter incorrect rows
        dataset = dataset.filter(lambda x: x["prompts"] is not None and x["labels"] is not None)

        # If provided, apply the postprocess_function to follow a specific prompt template
        if postprocess_function is not None: dataset = dataset.map(postprocess_function, num_proc=num_proc)

        # Join prompts and labels into "text". If we are in test mode, ignore labels for former's creation.
        dataset = dataset.map(create_text, fn_kwargs={"labels": include_labels_inside_text}, num_proc=num_proc)
        return dataset_support, dataset

    def get_support_and_tuning_dataset(
            self, 
            dataset_path: str, 
            split: str=None, 
            dataset_support: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]=None,
            postprocess_function=None,
            eos_token: str="</s>",
            include_labels_inside_text: bool=True,
            dynamic_examples: bool=False,
            num_examples: int=0, 
            shuffle: bool=False, 
            shuffle_seed: int=42, 
            num_proc: Union[int, None]=None
        ) -> Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None],
                   Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]]:
        # Load the dataset
        dataset = self._load_dataset(dataset_path, split)
        # Process the dataset
        return self._process_dataset(
            dataset=dataset,
            dataset_support=dataset_support,
            postprocess_function=postprocess_function,
            eos_token=eos_token,
            include_labels_inside_text=include_labels_inside_text,
            dynamic_examples=dynamic_examples,
            num_examples=num_examples,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            num_proc=num_proc
        )

    @property
    def dataset_recipe(self) -> DatasetRecipe:
        return self._dataset_recipe
