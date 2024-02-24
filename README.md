## Introduction
* One implementation of the paper __Multi-label and Multi-target Sampling of Machine Annotation for Computational Stance Detection__ in EMNLP 2023 <br>
* This repo and the enriched annotation are only for research use. Please cite the papers if they are helpful. <br>

## Package Requirements
* The scripts were tested on following libraries and versions:<br>
  python==3.9 <br>
  transformers==4.31 <br>
  torch==1.13.1 <br>

* To run the adversarial strategy of multi-target machine annotation, please install benepar for the parsing step:<br>
  https://github.com/nikitakit/self-attentive-parser

## Data Format
+ The original SemEval16 Tweet Stance Detection data are located at `./Stance_Data_JSON/`.<br>
+ The machine annotation of the SemEval16 Tweet Stance Detection (Task A) is located at `./Machine_Annotation/`.<br>
+ The stance label value 1 denotes '__Favor__', 0 denotes '__Against__', and 2 denotes '__None__'.<br>

## Running Machine Annotation
+ Run the `Model_as_Service.py` to start the LLM service and parsing model.<br>
+ Run the `Machine_Annotation_main.py` for your experiments on machine annotation for stance detection data.<br>
+ You can convert other stance detection corpora to json format.<br>
+ The default LLM is `vicuna-13b-v1.5`, you can change it in the `Machine_Annotation_main.py`. Please adjust the prompt format according to the LLM you use.<br>



## Citation
If the work is helpful, please cite our papers in your reports, slides, and papers.<br>

```
@inproceedings{liu-etal-2023-multi,
    title = "Multi-label and Multi-target Sampling of Machine Annotation for Computational Stance Detection",
    author = "Liu, Zhengyuan  and
      Chieu, Hai Leong  and
      Chen, Nancy",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.174",
    doi = "10.18653/v1/2023.findings-emnlp.174",
    pages = "2641--2649",
}
```
<br>

```
@inproceedings{liu-etal-2023-guiding,
    title = "Guiding Computational Stance Detection with Expanded Stance Triangle Framework",
    author = "Liu, Zhengyuan  and
      Yap, Yong Keong  and
      Chieu, Hai Leong  and
      Chen, Nancy",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.220",
    doi = "10.18653/v1/2023.acl-long.220",
    pages = "3987--4001",
}
```

