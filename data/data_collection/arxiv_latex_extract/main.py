#!/usr/bin/env python
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from os import sched_getaffinity
from os.path import basename, join, dirname, abspath
from shutil import copy
from tempfile import TemporaryDirectory
from typing import Callable

from tqdm import tqdm
from pathlib import Path
import yaml

from ale import LATEX_DIR
from ale.arxiv import delete, download
from ale.cleaner import ArxivCleaner

SCRIPT_DIR = dirname(abspath(__file__))
CFG_PATH = join(SCRIPT_DIR, "..", "collection_config.yaml")
_global_config = yaml.safe_load(open(CFG_PATH, "r"))

def clean(archive, output, target_dir=LATEX_DIR, filter_func=lambda _: True, verbose=False):
    # create temporary work directory
    with TemporaryDirectory() as work_dir:
        arxiv_cleaner = ArxivCleaner(
            data_dir=archive,
            work_dir=work_dir,
            target_dir=target_dir,
            filter_func=filter_func
        )

        return arxiv_cleaner.run(out_fname=output, verbose=verbose)

def process(archive, **kwargs):
    # if isinstance(archive, Callable): # handle lazy downloading
    #     archive = archive()

    # output = f"{basename(archive)}.jsonl"
    # path = clean(archive, output, **kwargs)
    # delete(archive)
    # return path

    try:
        if isinstance(archive, Callable):
            archive = archive()

        output = f"{basename(archive)}.jsonl"

        # resume from checkpoint
        if Path(output).exists():
            print(f"Skip {output}")
            return output
    
        path = clean(archive, output, **kwargs)
        if path is not None:
            delete(archive)

        return path

    except Exception as e:
        print(
            f"[ERROR] archive={archive} "
            f"type={type(e).__name__} "
            f"msg={e}"
        )
        return None


if __name__ == "__main__":
    def filter_func(tex): return b"tikzpicture" in tex # only process projects which contain tikz
    # cutoff = datetime(2005, 10, 23) # tikz 1.0 release date
    cutoff_str = _global_config.get("cutoff_date", "2026-04-01")
    cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d")
    exclude = _global_config.get("exclude_date", [])

    # Parallelize to make things faster
    num_workers = _global_config.get("num_worker", 8)
    # with Pool(num_workers:=len(sched_getaffinity(0))) as p:
    with Pool(num_workers) as p:
        print(f"Parallel processing on {num_workers} workers.")

        # save results in a tmpdir first and copy them into LATEX_DIR only when
        # processing is finished. This avoids partial files in case of errors
        with TemporaryDirectory() as target_dir:
            tasks = list(download(lazy=True, cutoff=cutoff, exclude=exclude))
            # tasks = list(download(lazy=True, cutoff=cutoff))
            kwargs = dict(filter_func=filter_func, target_dir=target_dir)

            for path in tqdm(p.imap_unordered(partial(process, **kwargs), tasks), total=len(tasks)):
                if path is None:
                    continue
                copy(path, LATEX_DIR)
