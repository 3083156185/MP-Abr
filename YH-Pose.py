import numpy as np
from collections import namedtuple
import cv2
from pathlib import Path

from sympy.physics.quantum.circuitplot import matplotlib

from FPS import FPS, now
import argparse
import os
from openvino.inference_engine import IENetwork, IECore
from Tracker import TrackerIoU, TrackerOKS, TRACK_COLORS
from IPython.display import Image
import pandas as pd
import numpy as np

# %matplotlib inline

DEFAULT_MODEL = "models/movenet_multipose_lightning_192x256_FP32.xml"

img_folder = 'images2'

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

LINES_BODY = [[4, 2], [2, 0], [0, 1], [1, 3],
              [10, 8], [8, 6], [6, 5], [5, 7], [7, 9],
              [6, 12], [12, 11], [11, 5],
              [12, 14], [14, 16], [11, 13], [13, 15]]


class Body:
    def __init__(self, score, xmin, ymin, xmax, ymax, keypoints_score, keypoints, keypoints_norm):
        self.score = score  # global score
        # xmin, ymin, xmax, ymax : bounding box
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoints_score = keypoints_score  # scores of the keypoints
        self.keypoints_norm = keypoints_norm  # keypoints normalized ([0,1]) coordinates (x,y) in the input image
        self.keypoints = keypoints  # keypoints coordinates (x,y) in pixels in the input image

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

    def str_bbox(self):
        return f"xmin={self.xmin} xmax={self.xmax} ymin={self.ymin} ymax={self.ymax}"


# Padding (all values are in pixel) :
# w (resp. h): horizontal (resp. vertical) padding on the source image to make its ratio same as Movenet model input.
#               The padding is done on one side (bottom or right) of the image.
# padded_w (resp. padded_h): width (resp. height) of the image after padding
Padding = namedtuple('Padding', ['w', 'h', 'padded_w', 'padded_h'])


# fall1.mp4 ==0  fight1.mp4==1 stand1.mp4==2  walk1.mp4==3
class MovenetMPOpenvino:
    def __init__(self, input_src="walk1.mp4",
                 xml=DEFAULT_MODEL,
                 device="CPU",
                 tracking=False,
                 score_thresh=0.2,
                 output='output.mp4'):

        self.score_thresh = score_thresh
        self.tracking = tracking
        if tracking is None:
            self.tracking = False
        elif tracking == "iou":
            self.tracking = True
            self.tracker = TrackerIoU()
        elif tracking == "oks":
            self.tracking = True
            self.tracker = TrackerOKS()

        if input_src.endswith('.jpg') or input_src.endswith('.png'):
            self.input_type = "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)

            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(self.img_w)
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video FPS:", self.video_fps)

        # Load Openvino models
        self.load_model(xml, device)

        # Rendering flags
        self.show_fps = True
        self.show_bounding_box = False

        if output is None:
            self.output = None
        else:
            if self.input_type == "image":
                # For an source image, we will output one image (and not a video) and exit
                self.output = output
            else:

                # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps,
                                              (self.img_w, self.img_h))

                # Define the padding
        # Note we don't center the source image. The padding is applied
        # on the bottom or right side. That simplifies a bit the calculation
        # when depadding
        if self.img_w / self.img_h > self.pd_w / self.pd_h:
            pad_h = int(self.img_w * self.pd_h / self.pd_w - self.img_h)
            self.padding = Padding(0, pad_h, self.img_w, self.img_h + pad_h)
        else:
            pad_w = int(self.img_h * self.pd_w / self.pd_h - self.img_w)
            self.padding = Padding(pad_w, 0, self.img_w + pad_w, self.img_h)
        print("Padding:", self.padding)

    def load_model(self, xml_path, device):

        print("Loading Inference Engine")
        self.ie = IECore()
        print("Device info:")
        versions = self.ie.get_versions(device)
        print("{}{}".format(" " * 8, device))
        print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[device].major, versions[device].minor))
        print("{}Build ........... {}".format(" " * 8, versions[device].build_number))

        name = os.path.splitext(xml_path)[0]
        bin_path = name + '.bin'
        print("Pose Detection model - Reading network files:\n\t{}\n\t{}".format(xml_path, bin_path))
        self.pd_net = self.ie.read_network(model=xml_path, weights=bin_path)
        # Input blob: input:0 - shape: [1, 3, 256, 256] (lightning)
        # Output blob: Identity - shape: [1, 6, 56]
        self.pd_input_blob = next(iter(self.pd_net.input_info))
        print(
            f"Input blob: {self.pd_input_blob} - shape: {self.pd_net.input_info[self.pd_input_blob].input_data.shape}")
        _, _, self.pd_h, self.pd_w = self.pd_net.input_info[self.pd_input_blob].input_data.shape
        for o in self.pd_net.outputs.keys():
            print(f"Output blob: {o} - shape: {self.pd_net.outputs[o].shape}")
        self.pd_kps = "Identity"
        print("Loading pose detection model into the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=device)

        self.infer_nb = 0
        self.infer_time_cumul = 0

    def pad_and_resize(self, frame):
        """ Pad and resize the image to prepare for the model input."""

        padded = cv2.copyMakeBorder(frame,
                                    0,
                                    self.padding.h,
                                    0,
                                    self.padding.w,
                                    cv2.BORDER_CONSTANT)

        padded = cv2.resize(padded, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)

        return padded

    def pd_postprocess(self, inference):
        result = np.squeeze(inference[self.pd_kps])  # 6x56
        bodies = []
        for i in range(6):
            kps = result[i][:51].reshape(17, -1)
            bbox = result[i][51:55].reshape(2, 2)
            score = result[i][55]

            if score > self.score_thresh:
                ymin, xmin, ymax, xmax = (bbox * [self.padding.padded_h, self.padding.padded_w]).flatten().astype(
                    np.int)
                kp_xy = kps[:, [1, 0]]
                keypoints = kp_xy * np.array([self.padding.padded_w, self.padding.padded_h])
                # add
                # if keypoints:

                # keypoints1 = keypoints.astype(np.int)
                # keypoints_norm1 = keypoints / np.array([self.img_w, self.img_h])
                # print(keypoints_norm1)
                # print(str(i)+"ddddd")

                body = Body(score=score, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                            keypoints_score=kps[:, 2],
                            keypoints=keypoints.astype(np.int),
                            keypoints_norm=keypoints / np.array([self.img_w, self.img_h]))

                bodies.append(body)

        return bodies

    def pd_render(self, frame, bodies):
        thickness = 3
        color_skeleton = (255, 180, 90)
        color_box = (0, 255, 255)
        for body in bodies:
            if self.tracking:
                color_skeleton = color_box = TRACK_COLORS[body.track_id % len(TRACK_COLORS)]

            lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if
                     body.keypoints_score[line[0]] > self.score_thresh and body.keypoints_score[
                         line[1]] > self.score_thresh]

            cv2.polylines(frame, lines, False, color_skeleton, 2, cv2.LINE_AA)

            for i, x_y in enumerate(body.keypoints):
                if body.keypoints_score[i] > self.score_thresh:
                    if i % 2 == 1:
                        color = (0, 255, 0)
                    elif i == 0:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

            if self.show_bounding_box:
                cv2.rectangle(frame, (body.xmin, body.ymin), (body.xmax, body.ymax), color_box, thickness)

            if self.tracking:
                # Display track_id at the center of the bounding box
                x = (body.xmin + body.xmax) // 2
                y = (body.ymin + body.ymax) // 2
                cv2.putText(frame, str(body.track_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 4, color_box, 3)

    def run(self):
        print('START:')

        incnt = 0
        count = 0
        frame_buffer = 0
        self.fps = FPS()
        nb_pd_inferences = 0
        glob_pd_rtrip_time = 0

        while True:

            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                count += 1
                incnt += 1

                if not ok:
                    break

            padded = self.pad_and_resize(frame)
            # cv2.imshow("Padded", padded)

            frame_nn = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)[None,]
            pd_rtrip_time = now()
            inference = self.pd_exec_net.infer(inputs={self.pd_input_blob: frame_nn})
            glob_pd_rtrip_time += now() - pd_rtrip_time
            bodies = self.pd_postprocess(inference)
            with open('trainx.txt', 'a') as f1:

                with open('trainy.txt', 'a') as f2:

                    if (incnt % 5 == 0):
                        for body in bodies:

                            if body:

                                frame_buffer += 1

                                list_float = list(np.ravel(body.keypoints_norm))

                                a_str = []
                                for num in list_float:
                                    a_str.append(str(num))
                                b_st = ",".join(a_str)

                                # print(b_st)
                                f1.writelines(b_st + '\n')

                                if frame_buffer == 15:
                                    # fall1.mp4 ==0  stand1.mp4==1 stand1.mp4==2  walk1.mp4==3
                                    f2.writelines("3" + '\n')
                                    frame_buffer = 0

            if self.tracking:
                bodies = self.tracker.apply(bodies, now())
            self.pd_render(frame, bodies)
            nb_pd_inferences += 1

            self.fps.update()
            self.show_fps = True

            if self.show_fps:
                self.fps.draw(frame, orig=(50, 50), size=1, color=(240, 180, 100))
            # cv2.imwrite(os.path.join(img_folder, f'{incnt:06d}.png'), frame)

            if count > 500 and incnt % 300 == 0:
                print(incnt)
                # filename='photo.jpg'
                # cv2.imwrite(filename, frame)

                # display(Image(filename))

            # cv2.imshow("Movenet", frame)

            if self.output:
                if self.input_type == "image":
                    cv2.imwrite(self.output, frame)
                    break
                else:
                    self.output.write(frame)

            # key = cv2.waitKey(1)
            # if key == ord('q') or key == 27:
            #    break
            # elif key == 32:
            # Pause on space bar
            #    cv2.waitKey(0)
            # elif key == ord('f'):
            # self.show_fps = True
            # elif key == ord('b'):
            #    self.show_bounding_box = not self.show_bounding_box

        # Print some stats
        if nb_pd_inferences > 1:
            global_fps, nb_frames = self.fps.get_global()

            print(f"FPS : {global_fps:.1f} f/s (# frames = {nb_frames})")
            print(f"# pose detection inferences : {nb_pd_inferences}")
            print(f"Pose detection round trip   : {glob_pd_rtrip_time / nb_pd_inferences * 1000:.1f} ms")

        if self.output and self.input_type != "image":
            # cv2.imwrite(os.path.join(img_folder, f'{incnt:06d}.png'), frame)

            self.output.write(frame)
            self.output.release()


if 1:
    xml = DEFAULT_MODEL
    # fall1.mp4 ==0  fight1.mp4==1 stand1.mp4==2  walk1.mp4==3
    pd = MovenetMPOpenvino(input_src="walk1.mp4",
                           xml=DEFAULT_MODEL,
                           device="CPU",
                           tracking=False,
                           score_thresh=0.2)
    pd.run()