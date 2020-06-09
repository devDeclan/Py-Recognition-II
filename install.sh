echo "🤖 updating repositories"
sudo apt-get -q update


echo "🤖 installing necessary packages"

echo "👾 installing python3"
sudo apt -q install python3 -y

echo "👾 installing unrar"
sudo apt -q install unrar -y

echo "👾 installing unzip"
sudo apt -q install unzip -y