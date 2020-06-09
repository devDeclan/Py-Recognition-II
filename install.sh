echo "🤖 updating repositories"
sudo apt-get -qq update


echo "🤖 installing necessary packages"

echo "👾 installing python3"
sudo apt -qq install python3 -y

echo "👾 installing unrar"
sudo apt -qq install unrar -y

echo "👾 installing unzip"
sudo apt -qq install unzip -y