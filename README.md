# Synthetic pretraining of foundational SMPL-X human mesh estimation model SMPLer-X



## Install
```bash
conda create -n smplerx python=3.8 -y
conda activate smplerx
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install -r requirements.txt

# install mmpose
cd main/transformer_utils
pip install -v -e .
cd ../..
```

## Inference 
- Place the video for inference under `SMPLer-X/demo/videos`
- Prepare the pretrained models to be used for inference under `SMPLer-X/pretrained_models`
- Prepare the mmdet pretrained model and config under `SMPLer-X/pretrained_models`
- Inference output will be saved in `SMPLer-X/demo/results`

```bash
cd main
sh slurm_inference.sh {VIDEO_FILE} {FORMAT} {FPS} {PRETRAINED_CKPT} 

# For inferencing test_video.mp4 (24FPS) with smpler_x_h32
sh slurm_inference.sh test_video mp4 24 smpler_x_h32

```
## 2D Smplx Overlay
We provide a lightweight visualization script for mesh overlay based on pyrender.
- Use ffmpeg to split video into images
- The visualization script takes inference results (see above) as the input.
```bash
ffmpeg -i {VIDEO_FILE} -f image2 -vf fps=30 \
        {SMPLERX INFERENCE DIR}/{VIDEO NAME (no extension)}/orig_img/%06d.jpg \
        -hide_banner  -loglevel error

cd main && python render.py \
            --data_path {SMPLERX INFERENCE DIR} --seq {VIDEO NAME} \
            --image_path {SMPLERX INFERENCE DIR}/{VIDEO NAME} \
            --render_biggest_person False
```


## Training
```bash
cd main
sh slurm_train.sh {JOB_NAME} {NUM_GPU} {CONFIG_FILE}

# For training SMPLer-X-H32 with 16 GPUS
sh slurm_train.sh smpler_x_h32 16 config_smpler_x_h32.py

```
- CONFIG_FILE is the file name under `SMPLer-X/main/config`
- Logs and checkpoints will be saved to `SMPLer-X/output/train_{JOB_NAME}_{DATE_TIME}`


## Testing
```bash
# To eval the model ../output/{TRAIN_OUTPUT_DIR}/model_dump/snapshot_{CKPT_ID}.pth.tar 
# with confing ../output/{TRAIN_OUTPUT_DIR}/code/config_base.py
cd main
sh slurm_test.sh {JOB_NAME} {NUM_GPU} {TRAIN_OUTPUT_DIR} {CKPT_ID}
```
- NUM_GPU = 1 is recommended for testing
- Logs and results  will be saved to `SMPLer-X/output/test_{JOB_NAME}_ep{CKPT_ID}_{TEST_DATSET}`


## References
- [SMPLer-X](https://github.com/caizhongang/SMPLer-X)
- [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE)
- [OSX](https://github.com/IDEA-Research/OSX)
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)
