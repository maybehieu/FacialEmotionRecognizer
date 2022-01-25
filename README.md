# FacialEmotionRegconizer

A simple emotion regconizer project using images classification I did in my journey taking on AI

## 1.Requirements
```sh
git clone https://github.com/maybehieu/FacialEmotionRegconizer.git
pip install -r requirements.txt
python generate_training_data.py -d (~PATH/dataset/) -fer (~PATH/fer2013.csv) -ferplus (~PATH/fer2013new.csv)
```
- Note: Re-order the label.csv file to /dataset/label/ according to the instruction. (Or you could change the code according to your file structure, it's very simple)
## 2. Training
```sh
python train_mobile.py # for mobilenetv2 implementation

python train_vgg.py # for custom vgg13 implementation
```

## 3. Real-time prediction
```sh
python frame_pred.py
```
## 4. References 
- [Arxiv] - Official paper on FER+ datasets
- [Guide] and [Code] - Ideas comes from here
- Mentors: [QuangTran] & [AnNguyen]

    [Arxiv]: <https://arxiv.org/abs/1608.01041>
    [Guide]: <https://medium.com/@reachraktim/emotion-recognition-on-the-fer-dataset-using-pytorch-835ce93d52a5>
    [Code]: <https://github.com/borarak/emotion-recognition-vgg13>
    [QuangTran]: <https://github.com/pewdspie24>
    [AnNguyen]: <https://github.com/NguyenTheAn>