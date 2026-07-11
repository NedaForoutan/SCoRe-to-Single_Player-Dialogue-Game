![Playpen Logo](./playpen-logo.png)

The Playpen is an extension of the [clemcore](https://github.com/clp-research/clemcore) framework. It contains everything you need to get started **training** and **evaluating** models in - or from - interaction. It was originally introduced in our [EMNLP 2025 paper](https://arxiv.org/abs/2504.08590) of the same title. The code for the experiments in the paper can be found [here](https://github.com/lm-playpen/playpen-paper-2025). What you see here has been further evolved considerably, to support a much wider range of experiments and to be much more user friendly.


# Get Started
## Set up the workspace

Clone the repository and switch into the workspace.
```bash
git clone https://github.com/lm-playpen/playpen.git && cd playpen
```

Set up the Python environment. Note: Playpen requires Python 3.10+.
```bash
python -m venv venv --system-site-packages && source venv/bin/activate
```

Install the requirements, which include the [clemcore](https://github.com/clp-research/clemcore) framework and the libraries to train and evaluate Huggingface models.
```bash
pip install -e .
```

Import then the [clembench](https://github.com/clp-research/clembench) repository in a directory of your choice and install its requirements. This will make the game environments available for training and evaluation.
```bash
git clone https://github.com/clp-research/clembench
pip install -r your/path/to/clembench/requirements.txt
```

In case the _clembench_ repository ha been cloned inside of the Playpen project root, games will be discovered automatically.
In case the cloning has been performed elsewhere, rename the file `game_registry.json.template` contained in the Playpen project root into `game_registry.json` and edit it to point to the _clembench_ path.

The file should be structured as follows:
```json
[
  {
    "benchmark_path": "your/path/to/clembench"
  }
]
```

To verify which games are available use:
```bash
clem list games
```

If you are interested in running the provided training examples, you should install TRL.
```bash
pip install '.[trl]'
```

Furthermore, if you want to run the prepared trainers in `examples/trl` with local huggingface models, you should install the `huggingface` extra. If you do not want this, you can still use the `trl` extra to run the trainers with remote models.
```bash
pip install 'clemcore[huggingface]'
```
Note: If you want to use the `transformers` library directly, you cannot use the playpen CLI to run the trainers, but you have to run your own scripts.

Now that everything is set up, you may follow the next steps to enter deeper into how to train and evaluate a model.
# Evaluate a model
## model_registry.json
In order to evaluate a model, the first thing we have to do is to register it into the `model_registry.json` file contained within this repository.
Mind that there is a file with the same name within the cloned _clembench_ folder. You may ignore that one.

Here is an example of model registry's entry for the Llama-3.1-Instruct model from Huggingface:
```json
{
  "model_name": "Llama-3.1-8B-Instruct",
  "backend": "huggingface_local",
  "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "release_date": "2024-07-23",
  "open_weight": true,
  "parameters": "8B",
  "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
  "context_size": "128k",
  "license": {
    "name": "Meta",
    "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
  },
  "model_config": {
    "requires_api_key": true,
    "premade_chat_template": true,
    "eos_to_cull": "<\\|eot_id\\|>"
  }
}
```
Here we are specifying fundamental information to help the engine to detect your model.
Most importantly, we are specifying a `model_name` for our model, the backend (in our case it is huggingface, indicated as `huggingface_local`) and the Huggingface model ID.
We are also specifying generation settings indicating whether a chat template associated to the model already exists (`premade_chat_template`: true), and the EOS token.

Note that specifying this token is of fundamental importance for the correct functioning of the benchmark.  
We also specify whether access to the model is gated and an API key is required under `requires_api_key`.

After registering a model, you may verify whether it is visible by the system by typing in your CLI the following command:
```bash
playpen list models
```

Further details regarding the other fields of a model entry are available in the [official documentation](https://github.com/clp-research/clemcore/blob/main/docs/model_backend_registry_readme.md).
## Registering an API key
To register an API key, copy or rename the file called `key.json.template` contained within the project's folder as `key.json`.

You may notice that there are several entries with the names of popular providers.
In the case you are interested in running Huggingface models, insert your key under `huggingface`:
```json
{
  ...
  "huggingface": {
    "api_key": "YOUR_API_KEY"
  },
  ...
 }
```
The mechanism is equivalent for other providers.
## Finally, the evaluation
To evaluate a model's gameplay performance on the `playpen-data` validation split, run the following command:

```bash
playpen eval <model-name> --suite all
```

where `<model-name>` should match your model's name as specified in the model registry. Connecting to the example above,
```bash
playpen eval Llama-3.1-8B-Instruct --suite all
```

This operation will produce a `<model-name>.val.json` file which contains two values:

1. **clemscore**: the average gameplay performance on the interactive benchmark games
2. **statscore**: the average performance on the static benchmark datasets

The file is by default located in a `playpen-eval/<timestamp>` folder.

Eventually, it is also possible to perform the evaluation separately on interactive and static benchmarks, or on single games.

To evaluate your model only on interactive games, run:
```bash
playpen eval Llama-3.1-8B-Instruct --suite clem
```
For static benchmarks:
```bash
playpen eval Llama-3.1-8B-Instruct --suite static
```

If you need to run the evaluation on specific benchmarks, e.g., for wordle, you can use the `-g` option of the eval command as follows:
```bash
playpen eval Llama-3.1-8B-Instruct -g wordle 
```
This also works with multiple games: 
```bash
playpen eval Llama-3.1-8B-Instruct -g wordle ifeval
```

You may also eventually rescore episodes contained in a specific folder by indicating the folder under `-r` and setting `--skip-gameplay` to avoid rerunning the games from scratch:
```bash
playpen eval Llama-3.1-8B-Instruct --skip_gameplay -r playpen-eval/2026-03-05T09-37-23/
```
The existing scores will be overwritten.
# Train a model

The Playpen does not make many assumptions over the libraries or logic you will implement for training your models.
This means that you may implement your own training script as you see fit, and use either supervised fine-tuning or RL approaches.

To provide some structure to your code, your training script may inherit from the `BasePlaypenTrainer` class.
It is defined [here](https://github.com/lm-playpen/playpen/blob/main/playpen/base.py) and it takes as arguments a `learner` model (i.e. the model to fine-tune) and optionally a `teacher` which may be helpful in an RL setup.

You may find all our examples in the [examples](https://github.com/lm-playpen/playpen/tree/main/examples) folder.

## Supervised Finetuning

Supervised fine-tuning (SFT) is known to help learning in interaction as it shifts the model's distribution towards the interactive data it will operate on.

In the context of clembench this means to let the model observe patterns of interaction which occur in various dialogue games.

### Training an Huggingface local model in the Playpen via TRL using Supervised Finetuning

Let's then run the basic trainer defined at `examples/trl/sft_trainer_simple.py` using the `Llama-3.1-8B-Instruct` model as a learning agent.
You may substitute this model with one of your choice.

```bash
playpen run examples/trl/sft_trainer_simple.py -l Llama-3.1-8B-Instruct 
```
> **Note:** a learning agent is defined using the argument -l <model_name>, with the <model_name> taken from the `model_registry.json` file.

The `playpen` CLI properly loads the huggingface model and runs the trainer code in the specified file.
After the execution is completed successfully, a directory named `models/sft/Llama-3.1-8B-Instruct` will appear
containing a sub-folder, e.g. `checkpoint-84` with the updated parameters of the model.

> **Note:** The trainer in the example above uses as a training set the train split available at the [playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data/viewer/interactions) repository containing dialogues obtained from models playing the games available in the version 2.0 of clembench.
> You may check the `examples/trl/sft_trainer_simple.py` source code for more implementation details.

### Evaluate the fine-tuned model

To evaluate the effectiveness of our SFT approach, we can now run the trained model again on the clembench.
As we have seen above, we will have first to create a new entry for our model in our local `model_registry.json` file pointing to the checkpoint folder generated by the training script.

```json
{
  "model_name": "Llama-3.1-8B-Instruct-sft",
  "backend": "huggingface_local",
  "huggingface_id": "models/sft/Llama-3.1-8B-Instruct/checkpoint-84",
  "release_date": "2024-09-04",
  "open_weight": true,
  "parameters": "135M",
  "languages": ["en"],
  "context_size": "2048",
  "license": {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0"
  },
  "model_config": {
    "premade_chat_template": true,
    "eos_to_cull": "<\\|im_end\\|>"
  }
}
```
> **Note:** under 'huggingface_id' it is possible to specify the ID of a model from the Huggingface Hub rather than its local folder.

After this, we can run the benchmark, but this time specifying as model name that of our model:
```bash
playpen eval Llama-3.1-8B-Instruct-sft
```

> **Note:** You can look up baseline performances of other models on the leaderboard: https://clembench.github.io/leaderboard.html.

## Parameter Efficient Fine-tuning (PEFT)

One of the approaches you may use to train your model when under computational constraints is that of using Parameter-Efficient Fine-tuning strategies such as LoRA.
We show you how to do it in Playpen using TRL and Llama-3.1-8B-Instruct as model with [this](https://github.com/lm-playpen/playpen/blob/main/examples/trl/sft_trainer_lora.py) example.

You may notice that it is virtually equivalent to that introduced in the previous section, with the only difference in the specific settings for PEFT added to the TRL's SFTTrainer Object.

Similarly to above we may run the training, this time specifying a different training script and model for our new use-case.
```bash
playpen run examples/trl/sft_trainer_lora.py -l Llama-3.1-8B-Instruct 
```

The `playpen` CLI properly loads the huggingface model and runs the trainer code in the specified file.
When the command finished successfully, then there will be a `models/sft+lora/Llama-3.1-8B-Instruct` directory
containing a checkpoint folder, e.g. `checkpoint-78` **containing only the adapter's parameters**.

To evaluate the LoRA fine-tuned model we register it in the local `modal_registry.json`,
especially indicating the base model on top of which to put the trained adapters under `peft_model` in the `model_config` as follows:
```json
{
  "model_name": "Llama-3.1-8B-Instruct-sft-lora",
  "backend": "huggingface_local",
  "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "release_date": "2024-07-23",
  "open_weight": true,
  "parameters": "8B",
  "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
  "context_size": "128k",
  "license": {
    "name": "Meta",
    "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
  },
  "model_config": {
    "peft_model": "models/sft+lora/llama3-8b/checkpoint-78",
    "requires_api_key": true,
    "premade_chat_template": true,
    "eos_to_cull": "<\\|eot_id\\|>"
  }
}
```
> **Note:** If you want to evaluate quantized models, then you can simply add `load_in_8bit: true` or `load_in_4bit: true`
> in the model_config section of the model spec. Alternatively, you can also directly load a quantized model from the
> huggingface hub by specifying the according huggingface_id.

Now we run the usual command for running the evaluation:
```bash
playpen eval Llama-3.1-8B-Instruct-sft-lora
```

## Reinforcement Learning

Reinforcement Learning is a very interesting approach for training models on interactive tasks. Compared to Supervised Finetuning, it has the advantage of having a model learning from direct experience rather than by imitating other models' gameplay.

Clemcore relies on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) to transform games into RL environments ready for training. 
OpenEnv adopts a Gymnasium-style API interface, and this should facilitate RL practitioners. We invite you however to have a look at the library.

This [notebook example](./examples/openenv/wordle-trl.ipynb) shows how to train a model with GRPO on the Wordle game with _clemcore_ and _playpen_. 
You will notice that you will have to define a custom agent and a customized rollout function to play in the game environments.
Playpen offers a base agent class (_ClemAgent_, which can be found at `playpen/agents/clem.py`). This class helps you by providing already means for collecting observations and returning their history. 
What you have to do is to define the `act` method, where you define how your agent should act given the observation history.

# Dataset Generation

In the SFT examples introduced above we make use of the canonical [playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data) split 
where we converted the interactions obtained from several models playing the games into a conversational dataset (the original source of the data can be found [here](https://github.com/clembench/clembench-runs/tree/main/v2.0).
In HF, the main property of a conversational dataset is that it contains samples which specify a list of `messages`.
These messages usually iterate on roles, that is, between a `user` and an `assistant`, and carry textual content.

Instead of using the data we provide, you may be interested in collected your own. All you have to do is running the _clemcore_ cli command:
```bash
clem run -g "{'benchmark':['2.0']}" -m Llama-3.1-8B-Instruct
```
This will create a `results` directory with the model's gameplay recorded in `interaction.json` files.
Be careful, this is different from `playpen eval` introduced above mainly because it produces a different folder structure. The results are expected to be the same.

To create a conversational dataset based on the interaction files, run the following command:
```bash
python3 examples/trl/data_utils.py <path-to>/results/
```

This will create in `examples/trl/results.jsonl` containing all interactions in form of a conversational dataset.
Furthermore, the script adds a `meta` annotation that informs about
`game`, `experiment`, `task_id`, `player_name`, `game_role`, `model` and `outcome`
which can be used for filtering the samples in the dataset.

Notably, the dataset contains samples of interaction from both perspectives of the 2-player games.
For example, for taboo the dataset contains the same episode, once from the perspective of the guesser and
once from that of the clue giver.

> **Note:** The default implementation of TRL for SFT only trains the model to predict the last `assistant` messages.
> All other messages are handled as a prefix or context for the prediction.

# TL;DR

### Evaluating a model

Register your model in `model_registry.json`, then run:
```bash
playpen eval Llama-3.1-8B-Instruct --suite all
```

This produces a `<model-name>.val.json` file with `clemscore` and `statscore` in a `playpen-eval/<timestamp>` folder.

### Running the SFT TRL example with Llama-3.1-8B-Instruct (local)

Run the basic SFT trainer example with a Llama-3.1-8B-Instruct learner (`-l`).
The model is optimized by imitating examples from the [playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data) dataset.

```bash
playpen run examples/trl/sft_trainer_simple.py -l Llama-3.1-8B-Instruct
```

This saves the model checkpoint under a newly created folder at `models/sft/Llama-3.1-8B-Instruct`.

### Running the SFT+LoRA TRL example with Llama-3.1-8B-Instruct (local)

Run the SFT+LoRA trainer example to fine-tune using parameter-efficient LoRA adapters.

```bash
playpen run examples/trl/sft_trainer_lora.py -l Llama-3.1-8B-Instruct
```

This saves only the adapter parameters under a newly created folder at `models/sft+lora/Llama-3.1-8B-Instruct`.

### Running the RL (GRPO) example

See the notebook at `./examples/openenv/wordle-trl.ipynb` for an example of training with GRPO on the Wordle game using _clemcore_ and _playpen_.

### Generating your own dataset

Run a model on the benchmark to collect gameplay interactions:
```bash
clem run -g "{'benchmark':['2.0']}" -m Llama-3.1-8B-Instruct
```

Then convert the recorded interactions into a conversational dataset:
```bash
python3 examples/trl/data_utils.py <path-to>/results/
```

This creates `examples/trl/results.jsonl` with all interactions including `meta` annotations for filtering.

