import re
import sys
from typing import List

def remove_latex_comments(tex: str) -> str:
    """
    Remove LaTeX comments safely.

    Rules:
    - % starts comment only if NOT escaped (\\%)
    - number of backslashes before % must be EVEN
    """

    result = []
    i = 0
    n = len(tex)

    while i < n:
        if tex[i] == "%":

            # count backslashes before %
            bs = 0
            j = i - 1
            while j >= 0 and tex[j] == "\\":
                bs += 1
                j -= 1

            # odd number of backslashes => escaped %
            if bs % 2 == 1:
                result.append("%")
                i += 1
                continue

            # otherwise it's comment → skip to end of line
            while i < n and tex[i] != "\n":
                i += 1

            continue

        result.append(tex[i])
        i += 1

    return "".join(result)

def skip_whitespace_and_comments(tex: str, start: int) -> int:
    """
    Skip whitespace and comments starting from index `start`.
    Returns the index of the first non-whitespace, non-comment character.
    """

    i = start
    n = len(tex)

    while i < n:
        if tex[i].isspace():
            i += 1
            continue

        if tex[i] == "%":
            backslash_count = 0
            k = i - 1
            while k >= 0 and tex[k] == "\\":
                backslash_count += 1
                k -= 1
            
            if backslash_count % 2 != 0:
                break
            
            # skip to end of line
            while i < n and tex[i] != "\n":
                i += 1
            continue

        break

    return i


def find_matching(tex: str, start: int, left: str, right: str) -> int:
    """
    Find the matching closing bracket.
    """

    if start >= len(tex) or tex[start] != left:
        return -1

    depth = 1
    i = start + 1

    while i < len(tex):
        i = skip_whitespace_and_comments(tex,i)
        if tex[i] == left:
            if i == 0 or tex[i - 1] != "\\":
                depth += 1
        elif tex[i] == right:
            if i == 0 or tex[i - 1] != "\\":
                depth -= 1

        if depth == 0:
            return i
        
        # print(f"Current character at position {i}: '{tex[i]}'")  # Debugging line
        # print(f"Current depth: {depth}")  # Debugging line
        i += 1

    return -1


def is_path_node(tex, pos):
    """
    pos point to the start of "node"
    """

    # prevent matching words like "anode"
    if pos > 0 and (tex[pos-1].isalnum() or tex[pos-1] == "_"):
        return False

    end = pos + 4

    if end < len(tex) and (tex[end].isalnum() or tex[end] == "_"):
        return False

    # skip whitespace after "node"
    end = skip_whitespace_and_comments(tex, end)

    # node after whitespace must be followed by either "(", [" or "{"
    return end < len(tex) and tex[end] in "([{"


def find_next_path_node(tex: str, start: int) -> int:
    """
    from start find the next path node. returns -1 if not found.
    """
    pos = start

    while True:
        pos = tex.find("node", pos)
        if pos == -1:
            return -1
        if is_path_node(tex, pos):
            return pos
        pos += 4


def replace_all_nodes(tex: str, replacement="Text") -> str:
    """
    Universal TikZ node cleaner. Replace
        \\node {...}
    and
        node {...}
    with
        {Text}
    while keeping every option/style unchanged.
    """

    result: List[str] = []
    i = 0
    n = len(tex)

    while i < n:

        # find the next key word "\node" or "node"
        node_pos = None

        p1 = tex.find("\\node", i)
        print(f"Found \\node at {p1}")  # Debugging line
        p2 = find_next_path_node(tex, i)
        print(f"Found node at {p2}")  # Debugging line

        candidates = [p for p in [p1, p2] if p != -1]

        if not candidates:
            result.append(tex[i:])
            break

        node_pos = min(candidates)
        # keep the text before the node
        result.append(tex[i:node_pos])
    
        # the start of the node content
        j = node_pos

        ################################################################
        # Step 1: skip node keyword and whitespace/comments
        ################################################################
        if tex[j] == "\\":
            j += len(r"\node")
        else:
            j += len("node")

        j = skip_whitespace_and_comments(tex, j)
        ################################################################
        # Step 2: skip optional [...] (...) and find node content {...}
        ################################################################    
        while j < n and tex[j] != "{":
            print(f"Checking character at position {j}: '{tex[j]}'")
            if tex[j] == "[":
                end_opt = find_matching(tex, j, "[", "]")
                if end_opt == -1:
                    break
                j = end_opt
                print("find optional [], skipping to", j, "content char:", tex[j])
            elif tex[j] == "(":
                end_opt = find_matching(tex, j, "(", ")")
                if end_opt == -1:
                    break
                j = end_opt
                print("find optional (), skipping to", j, "content char:", tex[j])
            j_after = skip_whitespace_and_comments(tex, j)
            print(f"After skipping whitespace/comments, j_after is {j_after}", "character:", tex[j_after])

            if j_after == j:
                j += 1
                print(f"No Skipping, now at position {j}", "character:", tex[j])
            else:
                j = j_after
                print(f"Skipping whitespace/comments, now at position {j}", "character:", tex[j])

        if j >= n:
            break
        
        print(f"Start checking character at position {j}: '{tex[j]}'")
        print("Cheking { at", tex[j-10:j+10])
        end = find_matching(tex, j, "{", "}") # j is the start of the node content
        print(f"Matching closing brace found at position {end}")  # Debugging line
        if end == -1:
            break
        # print(j)
        # print(j_after)
        ################################################################
        # Step 3: find semicolon ;
        ################################################################
        # k = end + 1
        # while k < n and tex[k] != ";":
        #     k += 1

        ################################################################
        # rebuild
        ################################################################
        #print(j)
        result.append(tex[node_pos:j])
        result.append("{")
        if skip_whitespace_and_comments(tex, j + 1) == end:
            result.append("}")
        else:
            result.append(replacement)
            result.append("}")
        # result.append(tex[end + 1:k])

        # if k < n:
        #     result.append(";")
        #     k += 1
        # i = k

        i = end + 1

    return "".join(result)

def process_latex(tex: str, mode: str) -> str:

    # 1. clean_all_text：only insert tikzset
    if mode == "clean_all_text":
        if "\\begin{tikzpicture}" in tex:
            tex = tex.replace(
                "\\begin{tikzpicture}",
                "\\tikzset{every node/.append style={text opacity=0}}\n\\begin{tikzpicture}"
            )
        return tex
    # 2. replace_all: replace node content with replacement text
    elif mode == "replace_all":
        return replace_all_nodes(tex=tex, replacement="Text")
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":

    input_file = "/usr/prakt/s0042/projects/data/benchmark_data/references/datikz-test_row_00000000.txt"
    output_file = "/usr/prakt/s0042/projects/test/datikz-test_replace_row_00000000.txt"
    mode = "replace_all"  # modes: clean_all_text | replace_all

    with open(input_file, "r", encoding="utf-8") as f:
        tex = f.read()

    new_tex = process_latex(tex, mode) # modes: clean_all_text | replace_all

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_tex)

    print(f"Done. Output saved to {output_file}")