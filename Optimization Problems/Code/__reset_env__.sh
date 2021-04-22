rm -rf pyenv
rm -rf env
rm -rf _env

python3 -m venv _env
source _env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt