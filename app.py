import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.title("Camera Calibration using OpenCV")

calibration_type = st.radio("Select Calibration Type", ["Single Camera Calibration", "Stereo Camera Calibration"])

chessboard_size = (11, 7)
square_size = 1.0
MIN_VALID_CALIBRATION_IMAGES = 3 

# Default image paths (adjust if your folder structure is different)
# left1 = "Im_L_1.png"
# left2 = "Im_L_2.png"
# left3 = "Im_L_3.png"
# left4 = "Im_L_4.png"
# left5 = "Im_L_5.png"
# left6 = "Im_L_6.png"

# right1 = "Im_R_1.png"
# right2 = "Im_R_2.png"
# right3 = "Im_R_3.png"
# right4 = "Im_R_4.png"
# right5 = "Im_R_5.png"
# right6 = "Im_R_6.png"

objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size


# SINGLE CAMERA CALIBRATION FUNCTION

def process_single_camera(images):
    objpoints = []
    imgpoints = []

    for i, image_file in enumerate(images):
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw corners and show on Streamlit
            cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Image {i+1} - Corners Detected", channels="RGB")
        else:
            st.warning(f"Chessboard NOT detected in image {i+1}")


    if len(objpoints) >= 2:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        st.subheader("Intrinsic Matrix (Camera Matrix)")
        st.text(mtx)
        st.subheader("Distortion Coefficients")
        st.text(dist)
        return True
    else:
        st.error("Not enough valid chessboard images detected.")
        return False



# STEREO CALIBRATION FUNCTION

def process_stereo_camera(left_images, right_images):
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    st.info(f"Processing {len(left_images)} left and {len(right_images)} right image(s) for stereo calibration...")

    # Read all image bytes once to avoid re-reading issues
    # These will be lists of np.uint8 arrays (raw file bytes)
    left_images_byte_data = [img.read() for img in left_images]
    right_images_byte_data = [img.read() for img in right_images]
    
    # These will store the decoded image data (np.frombuffer result) if needed elsewhere,
    # or we can decode directly from byte_data. Let's store the np.frombuffer results for consistency with original logic.
    left_images_decoded_data = [np.frombuffer(data, np.uint8) for data in left_images_byte_data]
    right_images_decoded_data = [np.frombuffer(data, np.uint8) for data in right_images_byte_data]

    gray_l_shape = None # To store shape from a valid gray image

    for i in range(min(len(left_images_decoded_data), len(right_images_decoded_data))):
        img_l = cv2.imdecode(left_images_decoded_data[i], cv2.IMREAD_COLOR)
        img_r = cv2.imdecode(right_images_decoded_data[i], cv2.IMREAD_COLOR)

        if img_l is None or img_r is None:
            st.warning(f"Could not decode image pair {i+1}. Skipping.")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        if gray_l_shape is None: # Get shape from the first valid image
            gray_l_shape = gray_l.shape[::-1]


        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

        if ret_l and ret_r:
            objpoints.append(objp)
            # Refine corner locations
            corners_l_refined = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), 
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners_r_refined = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints_left.append(corners_l_refined)
            imgpoints_right.append(corners_r_refined)

            cv2.drawChessboardCorners(img_l, chessboard_size, corners_l_refined, ret_l)
            cv2.drawChessboardCorners(img_r, chessboard_size, corners_r_refined, ret_r)
            st.image([cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)],
                     caption=[f"Left {i+1} - Corners 👍", f"Right {i+1} - Corners 👍"], width=250)
        else:
            caption_l = f"Left {i+1} - Chessboard " + ("👍" if ret_l else "👎")
            caption_r = f"Right {i+1} - Chessboard " + ("👍" if ret_r else "👎")
            st.image([cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)],
                     caption=[caption_l, caption_r], width=250)
            if not ret_l: st.warning(f"Chessboard NOT detected in Left image {i+1}")
            if not ret_r: st.warning(f"Chessboard NOT detected in Right image {i+1}")


    st.write(f"Successfully detected chessboard in {len(objpoints)} image pairs.")

    if len(objpoints) >= MIN_VALID_CALIBRATION_IMAGES:
        if gray_l_shape is None:
            st.error("Could not determine image dimensions for calibration (no valid image pairs processed).")
            return False

        # Calibrate each camera individually first
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, gray_l_shape, None, None)
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, gray_l_shape, None, None)
        
        # st.subheader("Individual Camera Calibration Results")
        # st.markdown("---")
        # col1_calib, col2_calib = st.columns(2)
        # with col1_calib:
        #     st.text("Left Camera Intrinsic Matrix:")
        #     st.json(mtx_l.tolist()) # Using st.json for better formatting of numpy arrays
        #     st.text("Left Camera Distortion Coefficients:")
        #     st.json(dist_l.tolist())
        # with col2_calib:
        #     st.text("Right Camera Intrinsic Matrix:")
        #     st.json(mtx_r.tolist())
        #     st.text("Right Camera Distortion Coefficients:")
        #     st.json(dist_r.tolist())
        # st.markdown("---")

        # # Stereo Calibration
 
        stereo_flags = cv2.CALIB_FIX_INTRINSIC
        # stereo_flags = cv2.CALIB_RATIONAL_MODEL # Can add for better distortion model if needed

        retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_l, dist_l, mtx_r, dist_r, gray_l_shape,
            flags=stereo_flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5) # More iterations for stereo
        )

        st.subheader("Stereo Calibration Results")
        st.text(f"Rotation Matrix (R):\n{R}")
        st.text(f"Translation Vector (T):\n{T}")
        st.text(f"Essential Matrix (E):\n{E}")
        st.text(f"Fundamental Matrix (F):\n{F}")

        # st.subheader("Stereo Calibration Results")
        # st.markdown("---")
        # st.text(f"Stereo Calibration RMS Error: {retval:.4f}")
        # st.text("Rotation Matrix (R) between cameras:")
        # st.json(R.tolist())
        # st.text("Translation Vector (T) between cameras:")
        # st.json(T.tolist())
        # st.text("Essential Matrix (E):")
        # st.json(E.tolist())
        # st.text("Fundamental Matrix (F):")
        # st.json(F.tolist())
        # st.markdown("---")

        # Stereo Rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, gray_l_shape, R, T, alpha=0 # alpha=0 crops to valid pixels
        )
        
      


        map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, gray_l_shape, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, gray_l_shape, cv2.CV_32FC1)

        # Use the first *original* stereo pair's data (already read into memory) to visualize rectification
        if not left_images_decoded_data or not right_images_decoded_data:
             st.error("No image data available for rectification visualization.")
             return False

        img_l_display = cv2.imdecode(left_images_decoded_data[0], cv2.IMREAD_COLOR)
        img_r_display = cv2.imdecode(right_images_decoded_data[0], cv2.IMREAD_COLOR)

        if img_l_display is None or img_r_display is None:
            st.error("Error decoding first image pair for stereo rectification display. Files might be corrupted.")
            return False
        
        rect_l = cv2.remap(img_l_display, map1x, map1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r_display, map2x, map2y, cv2.INTER_LINEAR)

        # Draw horizontal lines for rectification visualization
        def draw_lines(image, num_lines=20, color=(0, 255, 0)):
            img_copy = image.copy()
            step = img_copy.shape[0] // num_lines
            for i in range(step, img_copy.shape[0], step):
                cv2.line(img_copy, (0, i), (img_copy.shape[1], i), color, 1)
            return img_copy

        rect_l_lines = draw_lines(rect_l)
        rect_r_lines = draw_lines(rect_r)
        st.subheader("Rectified Images with Epipolar Lines")
        st.image([cv2.cvtColor(rect_l_lines, cv2.COLOR_BGR2RGB), cv2.cvtColor(rect_r_lines, cv2.COLOR_BGR2RGB)],
                 caption=["Left Rectified", "Right Rectified"], width=300)
        st.markdown("---")

        # Disparity Map
        gray_rect_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_rect_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        # StereoBM for disparity (can also use StereoSGBM for better results)
        num_disparities = 16 * 6 # Must be divisible by 16
        block_size = 11 # Must be odd
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
 
        disparity = stereo.compute(gray_rect_l, gray_rect_r).astype(np.float32) / 16.0
        
        # Normalize disparity for visualization
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        st.subheader("Disparity Map (Depth Visualization)")
        st.image(disparity_visual, clamp=True, channels="GRAY", caption="Disparity Map")
        st.markdown("---")

        # Optional: Reproject to 3D (Q matrix is from stereoRectify)
        # points_3D = cv2.reprojectImageTo3D(disparity, Q)
        # For visualization, one might use Open3D or other libraries.
        st.info("✔️ 3D points can be computed using the disparity map and Q matrix (not visualized here).")
        st.success("Stereo camera calibration and rectification successful! 🎉")
        return True
    else:
        st.error(f"Stereo calibration failed. Need at least {MIN_VALID_CALIBRATION_IMAGES} valid image pairs, but only found {len(objpoints)}.")
        return False



# Define default image paths
left_images_paths = [f"Im_L_{i}.png" for i in range(1, 7)]
right_images_paths = [f"Im_R_{i}.png" for i in range(1, 7)]

if calibration_type == "Single Camera Calibration":
    st.subheader("Single Camera Calibration")
    
    # Load and display left images
    st.write("Using the following 6 left camera images for calibration:")
    cols = st.columns(6)
    for i in range(6):
        with cols[i]:
            img = Image.open(left_images_paths[i])
            st.image(img, caption=f"Left {i+1}")
    if st.button("Calibrate Single Camera"):
        # Load images into memory as PIL Images
        single_images = [open(path, 'rb') for path in left_images_paths]
        process_single_camera(single_images)

elif calibration_type == "Stereo Camera Calibration":
    st.subheader("Stereo Camera Calibration")
    
    st.write("Using the following 6 left and 6 right camera images for stereo calibration:")

    # Display left and right images side-by-side using PIL
    for i in range(6):
        col1, col2 = st.columns(2)
        with col1:
            st.image(left_images_paths[i], caption=f"Left {i+1}")
        with col2:
            st.image(right_images_paths[i], caption=f"Right {i+1}")

    if st.button("Calibrate Stereo Cameras"):
        # Open as binary streams
        left_images = [open(path, 'rb') for path in left_images_paths]
        right_images = [open(path, 'rb') for path in right_images_paths]
        process_stereo_camera(left_images, right_images)
