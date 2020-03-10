from blazeface_pytorch.blazeface import BlazeFace
from face_cropper import Cropper
from face_extractor import FaceExtractor
from video_reader import VideoReader

frames_per_video = 50

facedet = Cropper(im_width=256, im_height=256)

# facedet = BlazeFace()
# facedet.load_weights('/home/lyan/PycharmProjects/Pytorch_Retinaface/blazeface_pytorch/blazeface.pth')
# facedet.load_anchors("/home/lyan/PycharmProjects/Pytorch_Retinaface/blazeface_pytorch/anchors.npy")
# _ = facedet.train(False)

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)

t = face_extractor.process_video('/home/lyan/PycharmProjects/Pytorch_Retinaface/aassnaulhq.mp4')
# print(t)
import matplotlib.pyplot as plt
plt.imshow(t[0]['faces'][0])
plt.show()