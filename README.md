# News Timeline Summarization

### Updates

### Datasets

All datasets used in our experiments are [available here](https://drive.google.com/drive/folders/1gDAF5QZyCWnF_hYKbxIzOyjT6MSkbQXu?usp=sharing), including:
* T17
* Crisis
* Entities
### Library installation
The `news-tls` library contains tools for loading TLS datasets and running TLS methods.
To install, run:
```
pip install -r requirements.txt
pip install -e .
```
[Tilse](https://github.com/smartschat/tilse) also needs to be installed for evaluation and some TLS-specific data classes.

### Loading a dataset
Check out [news_tls/explore_dataset.py](news_tls/explore_dataset.py) to see how to load the provided datasets.

### Running methods & evaluation
Check out [experiments here](experiments).

### Format & preprocess your own dataset
If you have a new dataset yourself and want to use preprocess it as the datasets above, check out the [preprocessing steps here](preprocessing).

