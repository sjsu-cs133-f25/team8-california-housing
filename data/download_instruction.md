# Download Instructions — California Housing

We do not commit data to Git.

# requires 'kaggle' installed and authenticated

kaggle datasets download -d camnugent/california-housing-prices -p data
unzip -o data/california-housing-prices.zip -d data
rm data/california-housing-prices.zip
