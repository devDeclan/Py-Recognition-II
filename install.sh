echo "🤖 updating repositories"
sudo apt-get -qq update


echo "🤖 installing necessary packages"

echo "👾 installing python3"
sudo apt -qq install python3 -y

echo "👾 installing unrar"
sudo apt -qq install unrar -y

echo "👾 installing unzip"
sudo apt -qq install unzip -y

echo "👾 installing wget"
sudo apt -qq install wget -y


echo "🤖 setting up python environment"

echo "👾 installing virtualenv"
python3 -m pip install --user virtualenv

echo "👾 creating virtual environment"
virtualenv venv
source venv/bin/activate
echo "❗ dont forget to terminate environment"
echo "run deactivate"

echo "👾 installing python requirements"
python -m pip install -r requirements.txt


echo "🤖 yaayyy you are all set up, happy hacking!!!"