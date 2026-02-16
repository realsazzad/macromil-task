## Sentiment Classification for Movie Reviews (Macromil Task)
Please run the main.ipynb notebook to view the details
````jupyter notebook````



### Docker Build Instruction
#### Prerequisites
- Docker installed
- A fine-tuned model directory present at `./distilbert-imdb-best/` (contains `config.json`, `model.safetensors`, tokenizer files, etc.)

#### Build
```bash
docker build -t imdb-sentiment-api .
```
#### Run
```bash
docker run --rm -p 8000:8000 imdb-sentiment-api
```