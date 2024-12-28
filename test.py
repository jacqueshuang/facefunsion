

#
import subprocess
import cv2
from pathlib import Path
def capture_frame_from_stream(stream_url, output_image_path):
	# 使用OpenCV读取和显示图像
	cap = cv2.VideoCapture(stream_url)
	if not cap.isOpened():
		print("无法打开RTMP流")
		return
	itx = 1
	while True:
	 ret, frame = cap.read()
	 if not ret:
		 print("无法读取帧")
		 break
	 # cv2.imread("{}captured_frame.jpg".format(itx),cv2.IMREAD_COLOR)
	 cv2.imshow('Live Stream', frame)
	 itx += 1
	 if cv2.waitKey(1) & 0xFF == ord('q'):
		 break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	rtmp_url = "https://d1--cn-gotcha07.bilivideo.com/live-bvc/553360/live_5649601_5756686.flv?expires=1735353369&len=0&oi=1885263197&pt=h5&qn=10000&trid=10001e1927ec9da78c7e255f45765a676f56&sigparams=cdn,expires,len,oi,pt,qn,trid&cdn=cn-gotcha07&sign=afa4e3d11aaa4135013c868a00f83565&site=1d89a295c21b6d70544658fe5113b9e9&free_type=0&mid=0&sche=ban&trace=32&isp=cu&rg=South&pv=Guangdong&origin_bitrate=518496&hot_cdn=0&p2p_type=-1&deploy_env=prod&pp=rtmp&source=puv3_onetier&info_source=origin&score=1&suffix=origin&sk=1f2f776a3847ad0eca20f051dfe6676b&sl=1&vd=bc&src=puv3&order=1"
	output_image_path = "captured_frame.jpg"
	capture_frame_from_stream(rtmp_url, output_image_path)
