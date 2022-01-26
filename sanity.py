import cv2

# cap = cv2.VideoCapture('/media/datadrive/flickr/FLICKR_5k/videos/10000130166.mp4')
# cap = cv2.VideoCapture('/media/datadrive/flickr/FLICKR_5k/videos/10008553263.mp4')
cap = cv2.VideoCapture('/media/datadrive/flickr/FLICKR_5k/videos/10007936344.mp4')

frames = []
success, image = cap.read()
while success:
    frames.append(image)
    success, image = cap.read()
cap.release()
if len(frames) <= 1:
    print("Error loading file!")