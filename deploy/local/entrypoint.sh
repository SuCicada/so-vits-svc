#!/bin/bash
set -e

/home/peng/miniconda3/bin/conda init bash
echo "Running pre-start command..."
#your-command-here
git config --global --add safe.directory /app
source /root/.bashrc
export PATH=/home/peng/miniconda3/bin:$PATH
#pip install --force-reinstall -U sumake
#make runuvicorn
exec "$@"
