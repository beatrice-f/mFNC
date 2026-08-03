# The Multilingual FrameNet Corpus
A collection of framenet corpus from different languages harmonized in a single corpora.

| Language | Corpus | Source |
| ------------- | ------------- | --- |
| Brazilian | [Frame2 Corpus](https://aclanthology.org/2024.lrec-main.655/) | [Github](https://github.com/FrameNetBrasil/frame-squared) |
| Chinese | [Chinese FrameNet](https://ieeexplore.ieee.org/document/1598752) | [Upon Request](https://tianchi.aliyun.com/dataset/149079?spm=a2c22.12281909.0.0.21373aedjFG7sS) |
| Dutch | [Dutch FrameNet](https://aclanthology.org/2020.lrec-1.387/) | [Website](https://web.archive.org/web/20260112195820/https://www.dutchframenet.nl/data-releases/) |
| English | [FrameNet 1.7 corpus](https://framenet.icsi.berkeley.edu/) | NLTK |
| French | [ASFALDA corpus](https://sites.google.com/site/anrasfalda/home) | [Github](https://github.com/kleag/french-framenet/) |
| German | [Salsa corpus](https://www.coli.uni-saarland.de/projects/salsa/) | [Upon Request](https://www.coli.uni-saarland.de/projects/salsa/) |
| Italian | [IFrame corpus](http://sag.art.uniroma2.it/iframe/doku.php?id=resources:pisa:manual_salsa_style) | [Link](http://sag.art.uniroma2.it/iframe/doku.php?id=resources:pisa:manual_salsa_style) |
| Korean | [Korean FrameNet](https://github.com/machinereading/koreanframenet) | [Github](https://github.com/machinereading/koreanframenet) |
| Latvian | [FullStack-LV](http://lrec-conf.org/workshops/lrec2018/W5/pdf/9_W5.pdf) | [Github](github.com/LUMII-AILab/FullStack) |
| Swedish | [SweFN](https://spraakbanken.gu.se/en/projects/swedish-framenet-swefn) | [SweFN Website](https://spraakbanken.gu.se/en/resources/swefn) |


To generate the datasets run

```bash
$ python data/generate_brazilian_framenet.py -i data/frame-squared/data/PT.jsonl -o data/datasets/frame-squared-pt
$ python data/generate_chinese_framenet.py -i data/CFN/ --o data/datasets/cfn
$ python data/generate_dutch_framenet.py -i data/dutchframenet/v1.2/ -o data/datasets/dutchframenet
$ python data/generate_english_framenet.py -o data/datasets/framenet_v17
$ python data/generate_french_framenet.py -i data/french-framenet/allanno/ -o data/datasets/asfalda
$ python data/generate_german_framenet.py -i data/salsa/salsa-corpora/ -o data/datasets/salsa
$ python data/generate_italian_framenet.py -i data/isst_iframe/isst_annotated.xml -o data/datasets/iframe
$ python data/generate_korean_framenet.py -i data/koreanframenet/data/1.2/ -o data/datasets/koreanframenet
$ python data/generate_latvian_framenet.py -i data/FullStack/FrameNet/eval -o data/datasets/latvianframenet
$ python data/generate_swedish_framenet.py -i data/swefn/swefn-ex.xml -o data/datasets/swefn
```

The scripts produce three distinct JSONL files `<source>.train.jsonl`, `<source>.valid.jsonl`, and `<source>.test.jsonl`. For each resource, the splits are automatically created in a balanced way. It is possible to use the flag `--filter-language-specifics` to remove language specific frames.
We share publicly available datasets (`data/datasets_original` mantains language specific frames while `data/datasets` only uses English frames). For the German and Chinese languages, retrieve the resources from their respective websites.

Each JSONL contains a list of valid JSON object of the form
```json
{
    "sentence": "Forest fires continue to rage in Spain", // the raw sentence
    "tokens": ["Forest", "fires", "continue", "to", "rage", "in", "Spain" ], // the tokenized sentence
    "language": "en", // the language of the document
    "metadata": { ... }, // dataset specific metadata are contained in this object
    "frames": [
        {
            "idxs": [1, 1], // the index of the span (wrt tokens) that activates this frame
            "activation": "fires", // the token that activates the frame
            "name": "Fire_burning", // the frame name
            "roles": [
                {
                    "name": "Fuel", // the role (frame element) name
                    "filler": "Forest", // the token that fills this role
                    "idxs": [0, 0] // the index of the span (wrt tokens) that fills this roles
                },
                ...
            ]
        },
        ...
    ]
}
```
