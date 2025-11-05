# Frame mAP
python eval.py \
        --cuda \
        -d jhmdb21 \
        -v yowo_v2_large \
        -bs 16 \
        -size 224 \
        --weight /datadisk/lx/results/yowov2/jhmdb21_base/jhmdb21/yowo_v2_large/yowo_v2_large_epoch_9.pth \
        --cal_frame_mAP \
        --root /datadisk/lx \
        -ct 0.005 \
