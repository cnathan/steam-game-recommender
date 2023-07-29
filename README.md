# Steam Game Recommender
This project is a personalized game recommendation system for the Steam platform. It focuses on user-game interactions and leverages machine learning techniques to provide accurate game recommendations.

## Dataset
The dataset used in this project contains over 37 million cleaned and preprocessed user recommendations (reviews) from the Steam Store. It also includes detailed information about games and add-ons. You can download the dataset from Kaggle.

### Context
The dataset represents a many-many relation between a game entity and a user entity. It does not contain any personal information about users on the Steam Platform. All user IDs have been anonymized.

### Content
The dataset consists of three main entities:

- games.csv: A table of games (or add-ons) information on ratings, pricing in US dollars $, release date, etc. Extra non-tabular details on games, such as descriptions and tags, are in a metadata file.
- users.csv: A table of user profiles' public information: the number of purchased products and reviews published.
- recommendations.csv: A table of user reviews: whether the user recommends a product.

### Acknowledgements
The dataset was collected from the Steam Official Store. All rights on the dataset thumbnail image belong to the Valve Corporation.

## Installation and Usage
To clone this repository and run the project, use the following bash commands:

```
bash
git clone https://github.com/yourusername/steam-game-recommender.git
cd steam-game-recommender
python main.py
```

## Contributing
If you'd like to contribute to this project, please submit a pull request.

## License
This project is licensed under the MIT License.
