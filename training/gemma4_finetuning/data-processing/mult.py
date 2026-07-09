from pathlib import Path

from pf_utils.tikz_rendering import render_tex_to_png, TikzRenderError


OUT_DIR = Path("/tmp/page_check_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


single_page_tex = r"""
\documentclass[tikz,border=2pt]{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}
\draw[thick] (0,0) circle (1);
\node at (0,0) {OK};
\end{tikzpicture}
\end{document}
"""


with open("/app/data-processing/test.txt") as f:
    multi_page_tex = f.read() 


def test_render(name: str, tex: str, create_ds: bool):
    out_path = OUT_DIR / f"{name}.png"
    metrics = {}

    print()
    print("=" * 80)
    print(f"Test: {name}")
    print(f"create_ds={create_ds}")

    try:
        render_tex_to_png(
            tex_code=tex,
            output_path=out_path,
            metrics=metrics,
            create_ds=create_ds,
        )

        print("[OK] rendered")
        print("output:", out_path)
        print("metrics:", metrics)

    except TikzRenderError as e:
        print("[REJECTED / FAILED]")
        print("error:", str(e))
        print("metrics:", e.metrics)

    except Exception as e:
        print("[UNEXPECTED ERROR]")
        print(type(e).__name__, e)


def main():
    # Sollte funktionieren
    test_render(
        name="single_page_create_ds_true",
        tex=single_page_tex,
        create_ds=True,
    )

    # Sollte wegen mehreren Seiten abgelehnt werden
    test_render(
        name="multi_page_create_ds_true",
        tex=multi_page_tex,
        create_ds=True,
    )

    # Sollte trotzdem rendern, weil create_ds=False
    test_render(
        name="multi_page_create_ds_false",
        tex=multi_page_tex,
        create_ds=False,
    )


if __name__ == "__main__":
    main()