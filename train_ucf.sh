# Train YOWOv2 on UCF24 dataset
python train.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_large \
        --root /datadisk/lx \
        --num_workers 4 \
        --eval_epoch 1 \
        --max_epoch 10 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 8 \
        -accu 16 \
        -K 32 \
        -ct 0.005 \
        --save_folder /datadisk/lx/results/yowov2/weights_yolov8_cbam/ \

