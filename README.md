# product-pricing

This repo contains inference code for the solution to [Product Pricing Challenge](https://retailvisionworkshop.github.io/pricing_challenge_2021/).

You can find more information in the [report](https://trax-geometry.s3.amazonaws.com/cvpr_challenge/cvpr2021/pricing_challenge_technical_reports/2nd_place_solution_to_Product_Pricing.pdf).


## Quick start

### Get models
[Yolov5-based detection model](https://drive.google.com/file/d/1Pdib6716ncvG6CLYwm3yw4dHqS1xAfyi/view?usp=sharing),
[price segmentation model](https://drive.google.com/file/d/1398Z-7OrBEVWl9KDUKCmgfDJsbmA3_WT/view?usp=sharing),
[OCR model](https://drive.google.com/file/d/1W6jm16hBiZGtXNU8n-5GdhxzSyQ7Td36/view?usp=sharing), [GNN model](https://drive.google.com/file/d/1QEkGXJpRmNdrtbjrTmPlkJvOxiy82j3n/view?usp=sharing).

Product embedding model might be taken from [the original repo](https://github.com/mingliangzhang2018/AiProducts-Challenge).

### Build


```
make build
make run
```

### Data

See the main challenge [page](https://retailvisionworkshop.github.io/pricing_challenge_2021/) for more information.

### Step-by-step inference

#### 1. Detect & recognize prices

```
python prod_pricing/predict_prices.py
```

#### 2. Get product embeddings using pretrained model

```
python prod_pricing/prod_embeddings.py
```

#### 3. Produce matching

```
python prod_pricing/matching/match_better.py
```

#### 4. You are breathtaking!
