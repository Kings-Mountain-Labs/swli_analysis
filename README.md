# swli_analysis


## Installation
Download [UV](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies (this is a lot easier for newbies managing dependencies, right now this is nvidia GPU specific, but I will update that later when GPU is actually being utilized to support AMD / Intel ? and Apple Metal)

Download the dependencies

```bash
uv sync
```

Run the analysis
```bash
uv run fft_interp.py
```

