## Analysis of Cleavage Time in Human Embryo
In this repo, we provide code for data analysis of age of human embryo and their parents in earlier stages of its cleavages.

## Installation
You need to have python >= 3.8 installed on your machine. 

1. Clone the repository:
   ```bash
   git clone [<repo-link>](https://github.com/maryammsd/Embryo.git)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Pleaes create a directory called tracking in the root of the project and copy the directories, including .csv and .emb files for each embryo separately. Next, you can run the below command to get different graphs including scatter plot, violion plot, etc. showing the dependency between the age of embryos and their parents provided in all the embryos in tracking directory.

```
python run.py
```

You can explore different functions in the `run.py` file to see different graphs implemented to depict the relationship between the embryos and their parents life time. 
