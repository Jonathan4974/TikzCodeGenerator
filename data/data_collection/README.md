# Generated new data from Arxiv

## 0. Create the environment

```bash
conda create -n arxiv_collection python=3.10 -y
```

```bash
conda activate arxiv_collection
```

Install full [TeX Live](https://www.tug.org/texlive),[ghostscript](https://www.ghostscript.com) and
[poppler](https://poppler.freedesktop.org)

Verify the installation by:

```bash
tex --version
gs --version
pdftoppm -v
pdfinfo -v
```
Install c dependency:

```bash
conda install -c conda-forge gobject-introspection cairo libxml2-devel libxml2 pygobject pycairo -y
```

Install the dependency of Datikz:

```bash
cd data/data_collection/datikz/
pip install -r requirements.txt
```

Install the dependency of arxiv_latex_extract

```bash
cd ../arxiv_latex_extract/
pip install -r requirements.txt
```

Make sure that [`latexpand`](https://gitlab.com/latexpand/latexpand) is installed and on your
`PATH` by:

```bash
which latexpand
```

Install other dependency:
```bash
pip install boto3
```

---

## 1. Generate AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and GITHUB_TOKEN

Replace with your own keys:

```bash
export AWS_ACCESS_KEY="AKIAVRYSHPMD7MOTA65M"
export AWS_SECRET_KEY="TmoB9o3udCdHNmDRvl07Smxvvo2Rqir+ofoT3xc1"
export GITHUB_TOKEN="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"
```

---

## 2. Excecute arxiv_latex_extract

```bash
cd /usr/prakt/s0030/projects/tikzcodegenerator/data/data_collection/arxiv_latex_extract/

python main.py
```

## 3. Excecute datikz to extract `.parquet` data from `extracted`

Adjust the `path` and `num_arxiv_workers` as needed

```bash
cd /usr/prakt/s0030/projects/tikzcodegenerator/data/data_collection/datikz/

python main.py --arxiv_files "/usr/prakt/s0042/projects/data/arxiv_TikZ_dataset/2510_2603/extracted" --size 384 --num_arxiv_workers 3 --num_compile_workers 3
```

---

## 4. TMUX commands
Start a new tmux:
```bash
tmux new -s data_collection
```

To exit: Ctrl-b, then d

to join a running tmux:

```bash
tmux attach -t data_collection
```

List running tmux:
```bash
tmux ls
```

Stop a tmux:
```bash
exit
```
or 
```bash
tmux kill-session -t data_collection
```

