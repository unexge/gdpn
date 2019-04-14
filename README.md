# gdpn

Generate Dota2 Player Names with Char-level RNN.

## Installing dependencies

```bash
$ pip install -r requirements.txt
```

## Training

```bash
$ cat ./data/names.txt
Voruj
MidOne
rtz
Abed
...
```

```bash
$ python train.py
(EPOCH 0) loss 4.55 checkpoint saved to ./data/model/checkpoint-0-4_55.tar
(EPOCH 1) loss 4.29 checkpoint saved to ./data/model/checkpoint-1-4_29.tar
...
```

## Sampling

```bash
$ python sample.py ./data/model/checkpoint-1-4_29.tar
ухетахон
Cojlasiu
-ХанаБан
Solanung
Stastome
THagetow
Gessdapl
...
```
