# OpenAI Server

FastAPI API wrapper for quantized LLMs.

_Designed to allow simpler interactions with local LLMs, and provide features like response caching._

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- A Unix-like environment
- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/) v3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [docker](https://www.docker.com/) (optional)

## Installation

1. [Create an SSH key](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) and clone this repository.

```
    git clone git@github.kcl.ac.uk:hi/openai-server.git
```

(Alternative) Clone this repository using HTTPs, supplying username and password:

```
    git clone https://github.kcl.ac.uk/hi/openai-server.git
```

2. Install dependencies:

```
    uv sync
```

This installs everything needed to run the server against the Claude backend. 
To also support the quantized local models (see [Models](#models) below), add the `quant` extra instead:

```
    uv sync --extra quant
```

3. Create the output directory (the cache database itself is created automatically on first run):

```
    mkdir output
```

This step is only needed when running as a python package.
The Docker images store this in a named volume instead, created automatically by `docker compose up`.

### Models

Models currently supported by the server:

| Name | Hugging Face (HF) repository | HF filename | Notes |
| -------- | ------- | ------- | ------- |
| Llama 3.1 | SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF | meta-llama-3.1-8b-instruct.Q4_K_M.gguf | Quantized version of model not distributed by Meta directly. Model required for this server. |
| MedLlama  3 | johnsnowlabs/JSL-MedLlama-3-8B-v2.0 | N/A | Requires quantization via [llama.cpp](https://github.com/ggerganov/llama.cpp) before use. Model not required for this server. |
| Biomistral 3 | skfrost19/BioMistralMerged | biomistral-merged-v0.1.gguf | Quantized version of model not distributed directly by BioMistral. Model not required for this server. |

To download Llama 3.1 (required), and any of the other models, download using information in the table above, and note the download location for configuration:

```
    uvx --from huggingface-hub huggingface-cli download <HF repository> <HF filename> --local-dir .
```

### Configuration

1. Set environment variables in `.env`.
Current variables, also shown in [.env.template](.env.template), are:

| Variable | Details |
| -------- | ------- |
Llama__3_1__8B_Quant_Instruct | Location of downloaded Llama model (required) |
MedLlama__3__8B_Quant | Location of downloaded MedLama model (optional) |
Biomistral__7B_Quant | Location of downloaded Biomistral model (optional) |
MODEL_FOLDER | Parent folder of all models (e.g. `/home/user/models`, if `/home/user/models/[ModelA]` and `/home/user/models/[ModelB]` exist) (optional; required for Docker) | 

2. Determine suitable configuration options (`config/config.ini`):

| Option | Details | Default |
| ------ | ------- | ------- |
CACHE > ACTIVE | Whether to store prompt answers and return these to the same prompt in the future. Sourced from a DB rather than the LLM directly. | False |

## Testing

### Unit tests

[tox](https://tox.wiki/en/4.20.0/) is used a test orchestrator, creating environments for linting ([flake8](https://flake8.pycqa.org/en/latest/)), type checks ([mypy](https://mypy.readthedocs.io/en/stable/)) and finally units tests ([pytest](https://docs.pytest.org/en/stable/)). It can be run using `uvx tox`.
 A [Makefile](Makefile) has been included packaging common commands for convenience. `make test` runs a loop that will fail if any of the environments fail, providing easier to read output.

## Running 

### Python package

Run locally as a python package (e.g. for integration tests) as follows:

```
uv run openaiserver
```

### Docker

Two images are available.
[Dockerfile](Dockerfile) additionally installs the `quant` extra, for use with the quantized local models.
[Dockerfile.lite](Dockerfile.lite) omits it, for a smaller, faster build.

Run through docker as follows (either locally or remotely):

```
docker compose build app
docker compose up -d app
```

This starts both the `app` (full) and `app-lite` services.
The app can then be interacted with in the same manner as if running as a python package.

## Example server interaction

1. Install the OpenAI client:

```
    pip install openai
```

2. Create and run a python file containing the following code:

```python
    import os
    from openai import OpenAI

    client = OpenAI(
        base_url = 'http://localhost:8080/v1/',
        api_key = 'foo'
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model='Llama__3_1__8B_Quant_Instruct',
        max_tokens=1024,
        temperature=0.7
    )
```

(Alternative) Test using `curl`:

```
    curl http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Llama__3_1__8B_Quant_Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "Say this is a test"
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }'
```

## Editing

As this is a Python package, most of the logic is contained within the [src](src) folder.
General recommendations for editing are:

* Use `make prettier`, another command made available within the [Makefile](Makefile) for convenience, to automatically format code.
* Always runs tests (`make test`) before committing.
* Commits can be made as follows:

```
    git add .
    git commit -m "[details of changes]"
    git push
```

## Built With

* [FastAPI](https://fastapi.tiangolo.com/)
* [SQLAlchemy](https://www.sqlalchemy.org/)
* [llama.cpp python](https://github.com/abetlen/llama-cpp-python)

## Authors

* [kclhi](https://kclhi.org)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.