#!/bin/bash
set -euo pipefail

I9_USER="${I9_USER:-s0030}"

PRAKT_DIR="/usr/prakt/$I9_USER"
TOOLS_DIR="$PRAKT_DIR/projects/full_latex"
TEXLIVE_DIR="$TOOLS_DIR/texlive/2026"
TEXLIVE_USER_DIR="$TOOLS_DIR/texlive-user"

mkdir -p "$TOOLS_DIR"
cd "$TOOLS_DIR"

wget -qO- https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
  | tar xzf -

cd install-tl-*

cat > texlive.profile <<EOF
selected_scheme scheme-full
TEXDIR $TEXLIVE_DIR
TEXMFCONFIG $TEXLIVE_USER_DIR/texmf-config
TEXMFHOME $TEXLIVE_USER_DIR/texmf-home
TEXMFLOCAL $TOOLS_DIR/texlive/texmf-local
TEXMFSYSCONFIG $TEXLIVE_DIR/texmf-config
TEXMFSYSVAR $TEXLIVE_DIR/texmf-var
TEXMFVAR $TEXLIVE_USER_DIR/texmf-var
option_doc 0
option_src 0
EOF

./install-tl -profile texlive.profile

export PATH="$TEXLIVE_DIR/bin/x86_64-linux:$PATH"

which pdflatex
which kpsewhich
which dvisvgm
kpsewhich tikz.sty
kpsewhich pgf.sty

echo "Done."
