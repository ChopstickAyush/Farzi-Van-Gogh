import argparse
import torch


from painter import *

# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
parser.add_argument('--video_path', type=str, default='./test_videos/sunflowers.jpg', metavar='str',
                    help='path to test image (default: ./test_videos/sunflowers.jpg)')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str',
                    help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str',
                    help='size of the canvas for stroke rendering')
parser.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                    help='keep input aspect ratio when saving outputs')
parser.add_argument('--max_m_strokes', type=int, default=500, metavar='str',
                    help='max number of strokes (default 500)')
parser.add_argument('--m_grid', type=int, default=5, metavar='N',
                    help='divide an image to m_grid x m_grid patches (default 5)')
parser.add_argument('--beta_L1', type=float, default=1.0,
                    help='weight for L1 loss (default: 1.0)')
parser.add_argument('--with_ot_loss', action='store_true', default=False,
                    help='imporve the convergence by using optimal transportation loss')
parser.add_argument('--beta_ot', type=float, default=0.1,
                    help='weight for optimal transportation loss (default: 0.1)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net-light', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, zou-fusion-net, '
                         'or zou-fusion-net-light (default: zou-fusion-net-light)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_oilpaintbrush_light', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush_light)')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str',
                    help='dir to save painting results (default: ./output)')
parser.add_argument('--disable_preview', action='store_true', default=False,
                    help='disable cv2.imshow, for running remotely without x-display')
args = parser.parse_args()


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    



if __name__ == '__main__':

    pt = VideoPainter(args=args)
    pt._draw_frame_by_frame()

