# From Grad to GPT

My learning journey through Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), from gradients and backpropagation to building and training GPT.

This repository will contain my implementations, lecture notes, exercises, and experiments. My goal is to understand each component well enough to explain it and implement it myself.

## Repository Structure

The structure follows the ten videos in the playlist. The directories and the first micrograd notebook template are in place; the remaining files below will be added as I work through the course.

```text
from-grad-to-gpt/
├── notebooks/
│   ├── 01-micrograd.ipynb
│   ├── 02-makemore-bigram.ipynb
│   ├── 03-makemore-mlp.ipynb
│   ├── 04-activations-gradients-batchnorm.ipynb
│   ├── 05-backprop-ninja.ipynb
│   ├── 06-makemore-wavenet.ipynb
│   ├── 07-build-gpt.ipynb
│   └── 09-gpt-tokenizer.ipynb
├── notes/
│   └── 08-state-of-gpt.md
├── projects/
│   └── 10-gpt2/
│       ├── README.md
│       ├── train.py
│       ├── evaluate.py
│       └── generate.py
├── .gitignore
├── environment.yml
├── environment.resolved.yml          # Earlier Linux/WSL snapshot
├── environment.windows.resolved.yml  # Generated manually on Windows
├── LICENSE
└── README.md
```

- **`notebooks/`** — One main notebook per coding lesson, with explanations, implementations, exercises, and small experiments.
- **`notes/`** — Written notes for the State of GPT lecture.
- **`projects/10-gpt2/`** — Python scripts for the final GPT-2 reproduction, with setup and training instructions in its own README.

Numbering matches the YouTube playlist, including **State of GPT** as video 08. Each makemore notebook preserves that lesson's version so I can compare how the model develops.

Datasets, model checkpoints, and training logs will stay out of Git. Dataset sources and preparation steps will be documented alongside the code that uses them.

## Learning Progress

- [ ] 01 — Build micrograd: neural networks and backpropagation
- [ ] 02 — Build makemore: a bigram language model
- [ ] 03 — Build makemore, Part 2: multilayer perceptron
- [ ] 04 — Build makemore, Part 3: activations, gradients, and BatchNorm
- [ ] 05 — Build makemore, Part 4: manual backpropagation
- [ ] 06 — Build makemore, Part 5: a WaveNet-like architecture
- [ ] 07 — Build GPT from scratch
- [ ] 08 — State of GPT: training and using GPT assistants
- [ ] 09 — Build the GPT tokenizer
- [ ] 10 — Reproduce GPT-2 (124M)

## Notebook Format

Each notebook will follow the same basic structure:

1. **Learning goals** — The questions I want to answer.
2. **Implementation** — Code developed while working through the lecture, with explanations in my own words.
3. **Exercises and experiments** — Changes I try, results I observe, and what those results suggest.
4. **Takeaways and open questions** — What I understand and what I need to revisit.

## Getting Started

The course assumes familiarity with Python and basic mathematics, including derivatives. The setup below uses **Windows, Miniconda, and VS Code**. Run terminal commands in **Anaconda Prompt (cmd.exe)**, opened from the Windows Start menu.

Open the repository directory:

```bat
cd /d C:\Projects\from-grad-to-gpt
```

If you cloned the repository elsewhere, use that location instead. Activating a Conda environment does not change the current directory.

### Set up the environment

For a new Windows setup, create and activate the environment:

```bat
conda env create -f environment.yml
conda activate from-grad-to-gpt
```

If the Windows environment already exists, activate it:

```bat
conda activate from-grad-to-gpt
```

[environment.yml](environment.yml) pins the direct dependencies: Python 3.12.16, JupyterLab 4.6.6, ipykernel 7.4.0, NumPy 2.5.5, and Matplotlib 3.11.2. Conda installs Python and pip from conda-forge, then installs the notebook packages from PyPI using the file's `pip:` section. This uses the Windows wheels for NumPy and Matplotlib, avoiding their dependency on Conda's external numerical and rendering DLL paths. Transitive dependencies and build identifiers are not pinned in this file, so it is not a complete lockfile. A separate pip `requirements.txt` is not maintained.

Use `environment.yml` for the initial Windows installation. The existing `environment.resolved.yml` was exported from Linux/WSL and includes Linux-specific dependencies; it is retained as a record of that environment.

### Open the first notebook in VS Code

From the activated environment and repository directory, open a local VS Code window:

```bat
code --new-window .
```

1. Confirm the window is connected to your local Windows machine: the status bar should not show `WSL: Ubuntu`.
2. Install or enable Microsoft's **Python** and **Jupyter** extensions locally.
3. Open [01 — Building micrograd](notebooks/01-micrograd.ipynb).
4. Click **Select Kernel** at the top right. Choose **Select Another Kernel...** if shown, then **Python Environments...**, then **from-grad-to-gpt**.

The workspace settings select the Windows interpreter and provide a **Conda (from-grad-to-gpt)** terminal profile. These paths assume Miniconda is installed under `%USERPROFILE%\miniconda3`; adjust `.vscode/settings.json` if yours is elsewhere. New terminals use this profile instead of WSL. You can also open the repository through GitHub Desktop and select the same notebook kernel.

Fill in the TODO cells while working through the lecture. The template includes learning goals, implementation prompts, experiments, and a review checklist. Its starter cell uses only Python's standard library; NumPy and Matplotlib are available for later experiments. PyTorch will be added when a lesson needs it.

Confirm the notebook is using the intended environment by running this in a code cell:

```python
import sys
import numpy as np

print(sys.executable)
print(np.__version__)
```

Run the cell with **Shift+Enter**. With this checkout's Miniconda installation, the Python path should be `C:\Users\alext\miniconda3\envs\from-grad-to-gpt\python.exe`, and NumPy should report `2.5.5`. The installation path may differ on another computer. See the [VS Code kernel guide](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) for details.

Alternatively, run `jupyter lab` in the activated Anaconda Prompt to use notebooks in a browser.

### Manage dependencies

When a lesson needs another Python package, add it under `pip:` in `environment.yml` and apply the change yourself. After checking that it works, record its installed version with `package==version`. Change an existing pin deliberately when upgrading:

```bat
conda env update -n from-grad-to-gpt -f environment.yml
```

After installing or changing dependencies, check that the notebook still runs and refresh the Windows snapshot described below. Commit `environment.yml` and the updated Windows snapshot with the code that requires the change. Conda stores the environment itself outside this repository. See the [Conda environment guide](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) for more details.

### Export the Windows environment snapshot

The environment files serve different purposes:

| File | Purpose |
| --- | --- |
| `environment.yml` | A short, manually maintained list of direct dependencies pinned to checked versions. |
| `environment.windows.resolved.yml` | A snapshot generated on Windows, including installed versions and transitive dependencies. |
| `environment.resolved.yml` | The earlier Linux/WSL snapshot, retained separately. |

After installation, activate the environment and check the initial packages:

```bat
cd /d C:\Projects\from-grad-to-gpt
conda activate from-grad-to-gpt
python -c "import numpy, matplotlib, jupyterlab, ipykernel; print('Imports OK')"
```

From the repository root, export the installed versions:

```bat
conda env export -n from-grad-to-gpt --no-builds > environment.windows.resolved.yml
```

This creates or overwrites `environment.windows.resolved.yml`, leaving the earlier Linux snapshot intact. Run the export in Anaconda Prompt on Windows. The `--no-builds` option keeps package versions but omits Conda build identifiers. Do not add `--from-history` here: this snapshot should include indirect dependencies too.

The exported file includes both Conda dependencies and a `pip:` section for packages installed from PyPI; retain both sections when recreating the environment.

Open the generated file, remove the top-level `prefix:` line if present (it is a local installation path), and save it before committing. Regenerate this file from the environment after dependency changes rather than editing its version entries by hand.

### Recreate the recorded Windows environment

Once the snapshot has been exported, create a separate environment from it:

```bat
conda env create -n from-grad-to-gpt-reproduced -f environment.windows.resolved.yml
conda activate from-grad-to-gpt-reproduced
```

The snapshot records package versions but is not an exact build lockfile. It includes platform-specific packages, so use the Windows snapshot on Windows with the same architecture. For a different operating system, start from `environment.yml` and validate the resulting environment there.

Useful commands:

| Command | Purpose |
| --- | --- |
| `conda env list` | List environments; `*` marks the active one. |
| `conda activate from-grad-to-gpt` | Enter the project environment. |
| `conda list` | Inspect installed packages. |
| `conda deactivate` | Leave the current environment. |

## References and Acknowledgments

The course and its original teaching materials were created by **Andrej Karpathy**. This is an independent study repository. Implementations that follow or adapt his code will include attribution to the relevant source.

- [Course website](https://karpathy.ai/zero-to-hero.html)
- [YouTube playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Official course repository](https://github.com/karpathy/nn-zero-to-hero)
- [micrograd](https://github.com/karpathy/micrograd)
- [makemore](https://github.com/karpathy/makemore)
- [minbpe](https://github.com/karpathy/minbpe)
- [build-nanogpt](https://github.com/karpathy/build-nanogpt)

## License

My original contributions to this repository are licensed under the [MIT License](LICENSE).

Third-party code and materials remain subject to their respective licenses. Any included third-party code will retain the applicable copyright and license notices.
