# LLM from Scratch (Harry Potter Books)
```
LLM from Scratch (Harry Potter Books)
└── Step 1: Download Data
    └── harry-potter-books.zip
        └── Extract ➝ "02 Harry Potter and the Chamber of Secrets.txt"
            └── Rename ➝ harry-potter-02.txt

└── Step 2: Clean Data
    └── Run: python clean-llm-data.py harry-potter-02.txt
        └── Output ➝ cleaned_harry-potter-02.txt (488,771 characters)

└── Step 3: Train Model
    ├── Code:
    │   ├── GPTModel.py        # defines the model architecture
    │   └── train_loader.py    # loads & processes cleaned data
    └── Run: python train.py   # trains using cleaned_harry-potter-02.txt
```

## Download harry-potter-02.txt
```
$ curl -L -o harry-potter-books.zip \
  https://www.kaggle.com/api/v1/datasets/download/shubhammaindola/harry-potter-books
$ unzip harry-potter-books.zip
$ mv "02 Harry Potter and the Chamber of Secrets.txt" harry-potter-02.txt
``

## Clean data
```
$ python clean-llm-data.py harry-potter-02.txt
cleaned_harry-potter-02.txt 488771 characters
```

## Train data
```
# GPTModel.py
# train_loader.py # uses cleaned_harry-potter-02.txt
$ python train.py # uset GPTModel.py and train_loader.py
```
