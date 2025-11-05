#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

bridge = CvBridge()

depth_image = None
color_image = None

def depth_callback(msg):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

def color_callback(msg):
    global color_image
    color_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def main():
    rospy.init_node("rs_depth_viewer", anonymous=True)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, color_callback)

    rospy.loginfo("Press ESC to exit, click window to print depth value")

    while not rospy.is_shutdown():
        if depth_image is None or color_image is None:
            rospy.sleep(0.05)
            continue

        # 可視化
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("Color", color_image)
        cv2.imshow("Depth Raw", depth_colormap)

        # クリックで深度値出力
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                raw_val = depth_image[y, x]
                print(f"({x}, {y}) -> raw: {raw_val}")

        cv2.setMouseCallback("Depth Raw", on_mouse)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# # RealSenseノード起動
# roslaunch realsense2_camera rs_camera.launch
# rosrun your_package ros_d415_confirmation_depth.py
