#!/bin/bash
#SBATCH --job-name=env_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --time=0-00:10:00          # 2 hours for interactive demo
#SBATCH --partition=PRACT
#SBATCH --mail-type=NONE
#SBATCH --output=logs/check-%j.out
#SBATCH --error=logs/check-%j.err

echo "=== Checking for pdftoppm ==="
which pdftoppm
command -v pdftoppm
type -p pdftoppm

if ! command -v pdftoppm &>/dev/null; then
    echo "Not found in PATH, searching common directories..."
    ls -l /usr/bin/pdftoppm /usr/local/bin/pdftoppm /opt/*/bin/pdftoppm 2>/dev/null || true
fi

echo "=== PATH ==="
echo $PATH
echo "=== done ==="

echo "=== Checking for convert (ImageMagick) ==="
which convert