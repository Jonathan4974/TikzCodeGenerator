#!/bin/bash
set -euo pipefail

mkdir -p /usr/prakt/s0030/projects/tools
cd /usr/prakt/s0030/projects/tools

wget -qO- https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
  | tar xzf -

cd install-tl-*

cat > texlive.profile <<'EOF'
selected_scheme scheme-full
TEXDIR /usr/prakt/s0030/projects/tools/texlive/2026
TEXMFCONFIG /usr/prakt/s0030/projects/tools/texlive-user/texmf-config
TEXMFHOME /usr/prakt/s0030/projects/tools/texlive-user/texmf-home
TEXMFLOCAL /usr/prakt/s0030/projects/tools/texlive/texmf-local
TEXMFSYSCONFIG /usr/prakt/s0030/projects/tools/texlive/2026/texmf-config
TEXMFSYSVAR /usr/prakt/s0030/projects/tools/texlive/2026/texmf-var
TEXMFVAR /usr/prakt/s0030/projects/tools/texlive-user/texmf-var
option_doc 0
option_src 0
EOF

./install-tl -profile texlive.profile

export TEXLIVE_DIR=/usr/prakt/s0030/projects/tools/texlive/2026
export PATH="$TEXLIVE_DIR/bin/x86_64-linux:$PATH"

which pdflatex
which kpsewhich
which dvisvgm
kpsewhich tikz.sty
kpsewhich pgf.sty