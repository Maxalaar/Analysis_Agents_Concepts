import os

import cv2


def generate_video(images, output_video_path, fps=30):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    if len(images) == 0:
        print("The list of images is empty.")
        return

    height = images[0].shape[0]
    width = images[0].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path+'.mp4', fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("Error: Could not open the video writer.")
        return

    for image in images:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_image)

    video_writer.release()
