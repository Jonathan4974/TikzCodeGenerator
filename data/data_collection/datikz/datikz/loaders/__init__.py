import os
import json
from datetime import datetime
from contextlib import contextmanager
from datetime import datetime
from json import load
from urllib.request import Request, urlopen

from datasets import (
    disable_progress_bar,
    enable_progress_bar,
    is_progress_bar_enabled,
)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "repo_cache.json")

def get_creation_time(repo):
    # read local cache first to save request to github
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            if repo in cache:
                return datetime.strptime(cache[repo], "%Y-%m-%dT%H:%M:%SZ")
    else:
        cache = {}

    # request to github if no local cache found
    url = f"https://api.github.com/repos/{repo}"
    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (DaTikZ Dataset Loader)")
    
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")

    try:
        with urlopen(req, timeout=10) as response:
            data = load(response)
            created_at = data['created_at']
            
            # write in cache
            cache[repo] = created_at
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
                
            return datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        raise RuntimeError(
            f"\nFail to request creation time of repo [{repo}] from github\n"
            f"reason: {e}\n"
        ) from e
    # return datetime.strptime(load(urlopen(f"https://api.github.com/repos/{repo}"))['created_at'], "%Y-%m-%dT%H:%M:%SZ")

def lines_startwith(string, prefix):
    return all(line.startswith(prefix) for line in string.splitlines())

def lines_removeprefix(string, prefix):
    return "".join(line.removeprefix(prefix) for line in string.splitlines(keepends=True))

@contextmanager
def no_progress_bar():
    if is_progress_bar_enabled():
        try:
            yield disable_progress_bar()
        finally:
            enable_progress_bar()
