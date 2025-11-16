# BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval

## Quick Start with Synthetic Data

This repository includes synthetic legal case data for educational purposes located in the `data/` directory:
- `train_task2.json` - 30 paragraph pairs for BERT fine-tuning
- `valid_task2.json` - 6 paragraph pairs for validation
- `test_task1.json` - 10 documents for testing
- `test_paragraphs_processed_data.json` - Processed data for poolout conversion
- `task1_test_labels_2024.json` - Labels for evaluation

### Step-by-Step Execution:

**Stage 1: BM25 Selection** (calculate BM25 scores separately)

**Stage 2: Fine-tune BERT for paragraph pair classification**
```bash
uv run python train.py -c config/nlp/BertPoint.config -g 0
```

**Stage 3: Extract paragraph-level interactions**
```bash
uv run python poolout.py -c config/nlp/BertPoolOutMax.config -g 0 --checkpoint output/checkpoints/bert_finetuned/1.pkl --result output/results/pool_out_max.json
```

**Convert poolout results to training format**
```bash
uv run python poolout_to_train.py -in data/test_paragraphs_processed_data.json -out output/results/pool_out_max.json --result output/results/train_poolout.json
```

**Train Attention-based RNN (LSTM)**
```bash
uv run python train.py -c config/nlp/AttenLSTM.config -g 0
```

**Test with Attention-based RNN (GRU)**
```bash
uv run python test.py -c config/nlp/AttenGRU.config -g 0 --checkpoint output/checkpoints/attengru/59.pkl --result output/results/gru_results.json
```

**Parse and evaluate results**
```bash
uv run python parse_results.py
uv run python parse_results.py evaluate data/task1_test_labels_2024.json output/results/gru_parsed_result.json output/results/metrics.json
```

## Instructions to run in nvidia container

`NV_GPU=5,7 USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} nvidia-docker run -itd --rm --shm-size 5gb --name bert-pli -v ${PWD}:/app bert-pli:latest tail -f /dev/null`

This repository contains the code for BERT-PLI in our IJCAI-PRICAI 2020 paper: *BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval*. 

## File Outline

### Model

- ``./model/nlp/BertPoint.py``: model for Stage2: fine-tune a paragraph pair classification Task. 

- ``./model/nlp/BertPoolOutMax.py`` : model paragraph-level interactions between documents.

- ``./model/nlp/AttenRNN.py`` : aggregate paragraph-level representations. 

### Config

- ``./config/nlp/BertPoint.config`` : configuration of ``./model/nlp/BertPoint.py`` (Stage 2, fine-tune). 

- ``./config/nlp/BertPoolOutMax.config`` : configuration of ``./model/nlp/BertPoolOutMax.py`` .

- ``./config/nlp/AttenGRU.config`` / ``./config/nlp/AttenLSTM.config`` : configuration of ``./model/nlp/AttenRNN.py`` (GRU / LSTM, repectively)


### Formatter

- ``./formatter/nlp/BertPairTextFormatter.py``: prepare input for ``./model/nlp/BertPoint.py`` (Stage 2, fine-tune)

- ``./formatter/nlp/BertDocParaFormatter.py`` : prepare input for ``./model/nlp/BertPoolOutMax.py``

- ``./formatter/nlp/AttenRNNFormatter.py`` : prepare input for ``./model/nlp/AttenRNN.py`` 


### Examples

Examples of input data. Note that we cannot make the raw data public according to the memorandum we signed for the dataset. The examples here have been processed manually and differ from the true data.

- ``./examples/task2/data_sample.json``: example input for Stage 2 (fine-tune).

    The format:

```
    {
	"guid": "queryID_paraID",
	"text_a": text of the decision paragraph,
        "text_b": text of the candidate paragraph,
	"label": 0 or 1 
    }
```

- ``./examples/task1/case_para_sample.json``: example input used in ``./config/nlp/BertPoolOutMax.config``. 

    The format:

```
    {
	"guid": "queryID_docID",
	"q_paras": [...], // a list of paragraphs in query case,
	"c_paras": [...], // a list of parameters in candidate case,
	"label": 0, // 0 or 1, denote the relevance
    }
```

- ``./examples/task1/embedding_sample.json``: example input used in ``./config/nlp/AttenGRU.config`` and ``./config/nlp/AttenLSTM.config``

    The format:

```
    {
	"guid": "queryID_docID",
	"res": [[],...,[]], // N * 768, result of BertPoolOutMax,
	"label": 0, // 0 or 1, denote the relevance
    }
```

### Scripts

- ``poolout.py``/``train.py``/``test.py``, main entrance for *poolling out*, *training*, and *testing*.

### Requirements

- See ``requirements.txt``

## How to Run?

- Stage 1: BM25 Selection:

    The BM25 score is calculated according to the standard [scoring function](https://en.wikipedia.org/wiki/Okapi_BM25). We set $k_1=1.5$, $b=0.75$. 

- Stage 2: BERT Fine-tuning:

    ```bash
    uv run python train.py -c config/nlp/BertPoint.config -g [GPU_LIST]
    ```

- Stage 3: 

    Get paragraph-level interactions by BERT: 

    ```bash
    uv run python poolout.py -c config/nlp/BertPoolOutMax.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
    ```
    Train

    ```bash
    uv run python train.py -c config/nlp/AttenGRU.config -g [GPU_LIST] 

    uv run python train.py -c config/nlp/AttenLSTM.config -g [GPU_LIST]
    ```

    Test

    ```bash
    uv run python test.py -c config/nlp/AttenGRU.config -g [GPU_LIST] --checkpoint [path of model checkpoint] --result [path to save results] 

    uv run python test.py -c config/nlp/AttenLSTM.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
    ```


## Experimental Settings

### Data

**For Educational/Testing Purposes**: This repository includes synthetic legal case data in the `data/` directory that simulates the structure of COLIEE dataset:
- 30 paragraph pairs for training (balanced positive/negative examples)
- 6 paragraph pairs for validation
- 10 multi-paragraph documents for testing
- Realistic legal content covering various topics (contracts, constitutional law, civil procedure, etc.)

**For Research/Production**: Please visit [COLIEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/) to apply for the actual competition dataset.

Please email shaoyq18@mails.tsinghua.edu.cn for the checkpoint of fine-tuned BERT. 

### Evaluation Metric

We follow the evaluation metrics in [COLIEEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/). Note that results should be evaluated on the whole document pool (e.g., 200 candidate documents for each query case.)

$$ Precision = \dfrac{\textrm{correctly retrieved cases (paragraphs) for all queries}}{\textrm{retrieved cases (paragraphs) for all queries}} $$

$$ Recall = \dfrac{\textrm{correctly retrieved cases (paragraphs) for all queries}}{\textrm{relevant cases (paragraphs) for all queries}} $$ 

$$ F-measure = \dfrac{2 \times Precision \times Recall}{Precision + Recall} $$

### Parameter Settings

Please refer to the configuration files for parameters for each step. 

For example, in Stage 2, $learning rate = 10^{-5}$, $batch size = 16$, $training epoch = 3$. In Stage 3, $N=54$, $M=40$, $learning rate=10^{-4}$, $weight_decay=10^{-6}$.


## Contact
For more details, please refer to our paper [BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval](https://www.ijcai.org/Proceedings/2020/0484.pdf). 
If you have any questions, please email shaoyq18@mails.tsinghua.edu.cn . 


## Citation
```
@inproceedings{shao2020bert,
  title={BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval},
  author={Shao, Yunqiu and Mao, Jiaxin and Liu, Yiqun and Ma, Weizhi and Satoh, Ken and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20},
  pages={3501--3507},
  year={2020}
}
```
