# Stylized Neural Painting 
This approach is based on the paper [stylized-neural-painting](https://arxiv.org/abs/2011.08114). We build upon existing code from the official implementation  [here](https://github.com/jiupinjia/stylized-neural-painting).

The project was done as a part of the CS337- Artificial Intelligence and Machine Learning course at IIT Bombay, under professor Preeti Jyothi. The presentation that was done as a part of this project can be found [here](https://iitbacin-my.sharepoint.com/:p:/g/personal/210050029_iitb_ac_in/ESUe9KCGO9FKi8kL7hBTuykBrgOn9T-t6eNzZiJ4tlHgGg) . And the report can be found [here]().





## Requirements

See [Requirements.txt](Requirements.txt).




## Setup

1. Clone this repo:

```bash
git clone https://github.com/ChopstickAyush/Farzi-Van-Gogh.git
cd Farzi-Van-Gogh
```

2. Download one of the pretrained neural renderers from Google Drive (1. [oil-paint brush](https://drive.google.com/file/d/1sqWhgBKqaBJggl2A8sD1bLSq2_B1ScMG/view?usp=sharing), 2. [watercolor ink](https://drive.google.com/file/d/19Yrj15v9kHvWzkK9o_GSZtvQaJPmcRYQ/view?usp=sharing), 3. [marker pen](https://drive.google.com/file/d/1XsjncjlSdQh2dbZ3X1qf1M8pDc8GLbNy/view?usp=sharing), 4. [color tapes](https://drive.google.com/file/d/162ykmRX8TBGVRnJIof8NeqN7cuwwuzIF/view?usp=sharing)), and unzip them to the repo directory.

```bash
unzip checkpoints_G_oilpaintbrush.zip
unzip checkpoints_G_rectangle.zip
unzip checkpoints_G_markerpen.zip
unzip checkpoints_G_watercolor.zip
```


3. We have also provided some lightweight renderers where users can generate high-resolution paintings on their local machine  with limited GPU memory.  Please feel free to download and unzip them to your repo directory. (1. [oil-paint brush (lightweight)](https://drive.google.com/file/d/1kcXsx2nDF3b3ryYOwm3BjmfwET9lfFht/view?usp=sharing), 2. [watercolor ink (lightweight)](https://drive.google.com/file/d/1FoclmDOL6d1UT12-aCDwYMcXQKSK6IWA/view?usp=sharing), 3. [marker pen (lightweight)](https://drive.google.com/file/d/1pP99btR2XV3GtDHFXd8klpdQRSc0prLx/view?usp=sharing), 4. [color tapes (lightweight)](https://drive.google.com/file/d/1aHyc9ukObmCeaecs8o-a6p-SCjeKlvVZ/view?usp=sharing)).

```bash
unzip checkpoints_G_oilpaintbrush_light.zip
unzip checkpoints_G_rectangle_light.zip
unzip checkpoints_G_markerpen_light.zip
unzip checkpoints_G_watercolor_light.zip
```

4. You can also train your own renderer (see under section **To train your neural renderer**).

## Running Instructions

<img src=https://github.com/ChopstickAyush/Farzi-Van-Gogh/assets/96743541/53f8d431-6ec8-48b9-8676-eebc0187c40e width="400"/> <img src=https://github.com/ChopstickAyush/Farzi-Van-Gogh/assets/96743541/ba01fdb0-27c5-4299-adae-b837cae0239a width="400"/>

- Progressive rendering

```bash
python demo_prog.py --img_path ./test_images/apple.jpg --canvas_color 'white' --max_m_strokes 500 --max_divide 5 --renderer oilpaintbrush --renderer_checkpoint_dir checkpoints_G_oilpaintbrush --net_G zou-fusion-net
```

- Progressive rendering with lightweight renderer (with lower GPU memory consumption and faster speed)

```bash
python demo_prog.py --img_path ./test_images/apple.jpg --canvas_color 'white' --max_m_strokes 500 --max_divide 5 --renderer oilpaintbrush --renderer_checkpoint_dir checkpoints_G_oilpaintbrush_light --net_G zou-fusion-net-light
```

- Rendering directly from mxm image grids

```bash
python demo.py --img_path ./test_images/apple.jpg --canvas_color 'white' --max_m_strokes 500 --m_grid 5 --renderer oilpaintbrush --renderer_checkpoint_dir checkpoints_G_oilpaintbrush --net_G zou-fusion-net
```



- Rendering videos
[Watch Video](https://github.com/ChopstickAyush/Farzi-Van-Gogh/raw/main/output_images/output.mp4)
  
```bash
python demo_video.py --disable_preview --video_path ./sunset.mp4 --canvas_color 'white' --max_m_strokes 200 --m_grid 5 --renderer oilpaintbrush --renderer_checkpoint_dir checkpoints_G_oilpaintbrush_light --net_G zou-fusion-net-light --output_dir ./output
```

You can replace oilpaintbrush by markerpen, watercolor, and rectangle.


#### Style transfer

<img src=https://github.com/ChopstickAyush/Farzi-Van-Gogh/assets/96743541/a852b00e-51c5-46c1-99e7-12d9ceb270bf width="200"/> 
<img src=https://github.com/ChopstickAyush/Farzi-Van-Gogh/assets/96743541/e92d792d-133f-468a-88a9-8427ff33ce28 width="300"/>
<img src=https://github.com/ChopstickAyush/Farzi-Van-Gogh/assets/96743541/01cd1ac5-dc2b-4a44-bfcf-4d311a675a68 width="300"/> 

- First, you need to generate painting and save stroke parameters to output-dir

```bash
python demo.py --img_path ./test_images/sunflowers.jpg --canvas_color 'white' --max_m_strokes 500 --m_grid 5 --renderer oilpaintbrush --renderer_checkpoint_dir checkpoints_G_oilpaintbrush --net_G zou-fusion-net --output_dir ./output
```

- Then, choose a style image and run style transfer on the generated stroke parameters

```bash
python demo_nst.py --renderer oilpaintbrush --vector_file ./output/sunflowers_strokes.npz --style_img_path ./style_images/fire.jpg --content_img_path ./test_images/sunflowers.jpg --canvas_color 'white' --net_G zou-fusion-net --renderer_checkpoint_dir checkpoints_G_oilpaintbrush --transfer_mode 1
```

You may also specify the --transfer_mode (0: transfer color only, 1: transfer both color and texture)

## Running through SSH

If you would like to run remotely through ssh and do not have something like X-display installed, you will need --disable_preview to turn off cv2.imshow on the run.

```bash
python demo_prog.py --disable_preview
```


## To retrain your neural renderer

You can also choose a brush type and train the stroke renderer from scratch. The only thing to do is to run the following common. During the training, the ground truth strokes are generated on-the-fly, so you don't need to download any external dataset. 

```bash
python train_imitator.py --renderer oilpaintbrush --net_G light-net --checkpoint_dir ./checkpoints_G --vis_dir val_out --max_num_epochs 400 --lr 2e-4 --batch_size 64
```

Here you can select watercolor, markerpen, and rectangle in place of oilpaintbrush and also you can select net_g as plain-dcgan, dcgan-light, plain-unet, huang-net, zou-fusion-net, zou-fusion-net-light instead of light-net. light-net is our implementation that reduces the number of performances significantly from the original paper while still giving the same quality of output images.


