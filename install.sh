echo "🤖 updating repositories"
sudo apt-get --quiet update


echo "🤖 installing necessary packages"

echo "👾 installing python3"
sudo apt install --quiet python3 -y

echo "👾 installing unrar"
sudo apt install --quiet unrar -y

echo "👾 installing unzip"
sudo apt install --quiet unzip -y