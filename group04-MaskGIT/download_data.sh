cd SavedModels

# Download our pretrained MaskGIT model
echo "Downloading pretrained MaskGIT model"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CRrDcuAw-uM4T97j1na30XHuVOHdyExS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CRrDcuAw-uM4T97j1na30XHuVOHdyExS" -O epoch_22_model.pt && rm -rf /tmp/cookies.txt

cd ../VQGAN/pretrain

# Download pretrained VQGAN model
echo "Downloading pretrained VQGAN model"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19CKHdEviGGtfq_Upd3bHxF1QmLuXq9Th' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19CKHdEviGGtfq_Upd3bHxF1QmLuXq9Th" -O vqgan_imagenet_f16_1024.zip && rm -rf /tmp/cookies.txt

unzip vqgan_imagenet_f16_1024.zip
rm vqgan_imagenet_f16_1024.zip

cd ../../Data

echo "Downloading Imagenet64 Dataset"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eTLGce4YvdroJT4kBdz5ZV_Ik0cvPJZJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eTLGce4YvdroJT4kBdz5ZV_Ik0cvPJZJ" -O Imagenet64.zip && rm -rf /tmp/cookies.txt

unzip Imagenet64.zip
rm Imagenet64.zip
