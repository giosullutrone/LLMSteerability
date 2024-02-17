import transformers
from transformers import TrainingArguments
from typing import Dict, Union
from finetuners import FineTuner
from recipes import DatasetRecipe, PeftRecipe, ModelRecipe, ModelTemplateRecipe, TokenizerRecipe, QuantizationRecipe
from dispatchers import DatasetDispatcher, ModelDispatcher, PeftDispatcher, QuantizationDispatcher, TokenizerDispatcher
from utils import SystemTuning, fit_response_template_tokens, fit_system_template_tokens, get_config_from_argparser, most_common_words
from configs import ConfigTrain
from peft import PromptTuningConfig

import logging
logger = logging.getLogger("llm_steerability")


transformers.set_seed(42)


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ### Argparse
    # Here we create the config for training using the CLI.
    # Load the configuration instance using argparser
    config: ConfigTrain = get_config_from_argparser(ConfigTrain)
    # Set logger to debug if verbose is requested
    if config.verbose: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.INFO)
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Training Arguments definition
    # Here we create the training arguments that will be used for our training.
    # For more information, check out the huggingface documentation for "TrainingArguments"
    training_arguments: TrainingArguments = config.training_arguments
    logger.debug(f"Training Arguments used: {training_arguments}")
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Here we define the base model we want to train.
    # Note: can be both huggingface's path or a local path
    model_name: str = config.model_name
    logger.debug(f"Model name: {model_name}")
    # An output path where to save the final model
    #   Tip: normally at the end of the training tha weights saved are the latest but they often are not
    #   the best. To solve this in the "TrainingArguments" we can specify "load_best_model_at_end=True"
    #   So that at the end of the procedure the best weights (calculated using loss on validation set) are
    #   loaded and then saved.
    #   WARNING: In some situations the trainer may not be able to load the best model at the end
    #            [In the terminal "Could not locate the best model..." or similar will appear].
    #            In that case use save_total_limit
    output_path: str = training_arguments.output_dir
    logger.debug(f"Output path: {output_path}")
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Peft config initialization
    # For the fine-tuning we can use different peft adapters. 
    # Tip: Check out the PeftRecipe for the list of recipes available or create a custom one by extending the PeftRecipe class.
    # Here we initialize the adapter's configuration used in the following sections.
    peft_recipe: PeftRecipe = config.peft_recipe
    peft_config = PeftDispatcher(peft_recipe).get_peft_config() if peft_recipe is not None else None

    if peft_recipe is not None and peft_recipe.peft_config_obj == PromptTuningConfig and config.prompt_tuning_most_common_init:
        dataset_train = DatasetDispatcher(config.dataset_recipe).get_tuning_dataset(config.dataset_name, 
                                                                                    split="train",
                                                                                    eos_token="",
                                                                                    include_labels_inside_text=False)
        words = (" ".join(list(dataset_train["text"]))).lower().split(" ")
        peft_config.prompt_tuning_init_text = " ".join(most_common_words(words, peft_config.num_virtual_tokens))
        logger.debug(f"Prompt tuning init text set to: {peft_config.prompt_tuning_init_text}")
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Create tokenizer and model
    # Here we create the tokenizer for the specific model
    tokenizer_recipe: TokenizerRecipe = config.tokenizer_recipe
    tokenizer = TokenizerDispatcher(tokenizer_recipe).get_tokenizer(model_name)
    # Now we create the actual model
    model_recipe: ModelRecipe = config.model_recipe
    # We also initialize the quantization config (if provided)
    quantization_recipe: QuantizationRecipe = config.quantization_recipe
    quantization_config = QuantizationDispatcher(quantization_recipe).get_quantization_config() if quantization_recipe is not None else None
    # Note: during training, instead of providing the peft's config, you provide the path to the fine-tuned folder that will contain (if used) the trained adapter
    model = ModelDispatcher(model_recipe).get_model(model_name, quantization_config, peft_config)
    # --------------------------------------------------------------------------
    
    
    # --------------------------------------------------------------------------
    # ### Dataset
    # We load the datasets from huggingface using the specific recipe.
    # Tip: For more datasets check out recipes.DatasetRecipe, if you don't find the one that you want
    # follow the dataset recipe creation guide. 
    # TODO: Write a guide for the creation of new recipes
    # Tip: You can also use the YAML recipe to re-use any yaml config file available in lm-evaluation-harness library.
    #
    # Here we are creating a train and validation datasets with the given number of examples per prompt and
    # adapted to the template of the specific model:
    # ### Example of a prompt:
    # "What is the result of 2 + 2? Answer: "
    # -> for LLama2 we obtain "<SYS> </SYS> [INST] What is the result of 2 + 2? Answer: [/INST]"
    dataset_name: str = config.dataset_name
    dataset_recipe: DatasetRecipe = config.dataset_recipe
    model_template_recipe: ModelTemplateRecipe = config.model_template_recipe
    dataset_support, dataset_train = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                                      split="train", 
                                                                                                      num_examples=config.num_examples,
                                                                                                      postprocess_function=model_template_recipe.postprocess_function,
                                                                                                      eos_token=tokenizer.eos_token,
                                                                                                      include_labels_inside_text=True,
                                                                                                      num_proc=config.num_proc,
                                                                                                      dynamic_examples=config.dynamic_examples)
    try:
        _, dataset_val = DatasetDispatcher(dataset_recipe).get_support_and_tuning_dataset(dataset_name, 
                                                                                          split="validation", 
                                                                                          dataset_support=dataset_support,
                                                                                          num_examples=config.num_examples,
                                                                                          postprocess_function=model_template_recipe.postprocess_function, 
                                                                                          eos_token=tokenizer.eos_token,
                                                                                          include_labels_inside_text=True,
                                                                                          num_proc=config.num_proc,
                                                                                          dynamic_examples=config.dynamic_examples)
    except:
        # If you don't have a validation set available, split the training set
        assert config.validation_split_size is not None, "No validation set is available but validation split size has not been specified"
        logger.info("Validation dataset not found, splitting train into two different sets.")
        if config.dynamic_examples: logger.warning("Using dynamic examples with splitted sets currently leads to small amount of data leakage between training and validation set,\
                                                   use with caution as the validation set will likely overperform.")
        dataset = dataset_train.train_test_split(test_size=config.validation_split_size)
        dataset_train, dataset_val = dataset["train"], dataset["test"]

    if config.training_size: dataset_train = dataset_train.select(range(config.training_size))

    # Logger
    logger.debug(f'First prompt for training set:\n{dataset_train["text"][0]}')
    logger.debug(f'First prompt for validation set:\n{dataset_val["text"][0]}')
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### Response template
    # This is needed in case you want to train only on the completion of the input prompt.
    # ### Example of a prompt:
    # "What is the result of 2 + 2? Answer: 4"
    #                                      >-<
    # -> If the response template is "Answer: " then the loss will be calculated only on the following tokens in this case "4"
    #
    # Without a template, instead:
    # "What is the result of 2 + 2? Answer: 4"
    # >--------------------------------------<
    # -> The loss will be calculated on all tokens (i.e. we are training the model to also predict the question and not only the answer)
    # Here we are using an util function that given the specific dataset recipe and model recipe, finds the best response template
    # for the dataset selected and model selected.
    # [It uses the model recipe's template if it is available else the dataset recipe one and finds the best token ids to use for compatibility]
    if config.completion_only: 
        response_template = fit_response_template_tokens(dataset_train, dataset_recipe, model_template_recipe, tokenizer)
        logger.debug(f'Found the following response template tokens: {response_template}')
    else: response_template = None
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### System template
    # This is needed if you are using `prompt tuning` as PEFT and you want to start the prepended learned tokens after a specific
    # set of words i.e. the system template of the model.
    # For example:
    # LLama2 format: <s> ... <<SYS>>  <</SYS>> ...
    #                                ^
    #                                | 
    # In this position instead of at the start of the whole prompt (before even <s>)
    # 
    # This should make the tuning better mimic the performances obtained for the typical prompt engineering done by hand
    if peft_recipe is not None and peft_recipe.peft_config_obj == PromptTuningConfig and config.system_tuning:
        system_template = fit_system_template_tokens(dataset_train, model_template_recipe, tokenizer)
        model = SystemTuning.add_proxy(model, system_template=system_template, tokenizer=tokenizer)
        logger.debug(f'Added system tuning modification as requested')
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # ### FineTuner creation and training
    # We initialize the FineTuner class and provide all the arguments needed.
    finetuner = FineTuner(
        model=model,
        tokenizer=tokenizer,
        dataset_train=dataset_train,
        dataset_validation=dataset_val,
        response_template=response_template
    )
    # Note: we also specify other kwargs as they will be given to the underlying `SFTTrainer` init.
    # For more info on the available parameters, check out the documentation for trl's SFTTrainer.
    finetuner.train(output_path, training_arguments=training_arguments, **config.finetuner_arguments)

    # We set everything to None to free up memory before the creation of new models.
    # In this example we are training only one model/dataset therefore it is not really needed.
    finetuner = None
    model = None
    tokenizer = None
    quantization_recipe = None
    # --------------------------------------------------------------------------
