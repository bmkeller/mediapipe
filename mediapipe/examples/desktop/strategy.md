# Computer Vision Strategy: Mapping Camera Landmarks to Island and Projector Space

This document outlines the strategy for a computer vision system that maps landmarks detected in a camera's view to the normalized `[0,1]` coordinate systems of a kitchen island and an inset projector screen.

The core of this strategy is a **one-time calibration phase** followed by a lightweight, real-time processing loop. This approach ensures robustness against scene clutter and lighting changes, and maximizes performance.

---

## Phase 1: One-Time Calibration

The goal of this phase is to find the precise relationship between the camera's pixel space and the physical spaces of the island and projector. This is done once and the results are saved.

### 1.1. Island Boundary Detection (Markerless)

This procedure finds the four corners of the island without requiring physical markers. It should be run with a clear view of the island.

1.  **Image Pre-processing:**
    * Capture a high-resolution image from the camera.
    * Convert the image to **grayscale**.
    * Apply a `GaussianBlur` to reduce sensor noise and minor texture variations.

2.  **Edge Detection:**
    * Use the `Canny` edge detector to create a binary image highlighting sharp intensity changes. The boundary between the light island and dark floor should be a primary edge.

3.  **Contour Analysis:**
    * Use `findContours` to identify all closed shapes in the edge image. Use `RETR_EXTERNAL` to only find the outermost contours.
    * Find the contour with the **largest area** using `contourArea`. This will be the island.

4.  **Corner Extraction:**
    * On the largest contour, use `approxPolyDP` to simplify the shape into a polygon.
    * The key is to tune the `epsilon` parameter of this function until it reliably returns a shape with exactly **four vertices**. These four points are the island's corners in camera pixel coordinates.

### 1.2. Projector Boundary Detection (Automated)

This procedure uses a known pattern to find the projector's corners with sub-pixel accuracy.

1.  **Project Pattern:** Project a high-contrast **checkerboard or asymmetric circle grid** pattern to fill the entire projection area.
2.  **Detect Pattern:** Use OpenCV's built-in `findChessboardCorners` or `findCirclesGrid` function. These functions are highly optimized to find the pattern's corner points automatically.
3.  **Identify Boundary:** The four outermost points returned by the function correspond to the four corners of the projector area.

### 1.3. Compute and Store Transformations

For each set of four corner points (`p1, p2, p3, p4`):

1.  **Define Destination:** Create a destination set of four points representing the normalized `[0,1]` square: `(0,0), (1,0), (1,1), (0,1)`.
2.  **Compute Homography:** Use `getPerspectiveTransform` with the source (camera) points and destination (normalized) points.
3.  **Save Matrices:** This function returns a 3x3 perspective transformation matrix (`H`). Compute one for the island (`H_island`) and one for the projector (`H_projector`). Save these two matrices to a file (e.g., XML or YAML using OpenCV's `FileStorage`).

---

## Phase 2: Real-Time Landmark Mapping

This is the main operational loop of the application. It's designed to be extremely fast.

1.  **Load Matrices:** On startup, load the pre-computed `H_island` and `H_projector` matrices from the file.
2.  **Get Landmark:** For each landmark `(x,y)` detected by your primary vision algorithm in camera pixel coordinates:
3.  **Transform Point:** Use the `perspectiveTransform` function to map the landmark's coordinates using both matrices.
    * `transformed_island_point = perspectiveTransform(landmark_point, H_island)`
    * `transformed_projector_point = perspectiveTransform(landmark_point, H_projector)`
4.  **Validate and Use:** Check if the transformed coordinates fall within the normalized `[0,1]` range.
    * `if (0 <= transformed_island_point.x <= 1 && 0 <= transformed_island_point.y <= 1)`: The landmark is on the island.
    * `if (0 <= transformed_projector_point.x <= 1 && 0 <= transformed_projector_point.y <= 1)`: The landmark is on the projector.
    * If a point is outside this range, it is not within that respective area.

---

## Phase 3: System Maintenance (Periodic Validation)

This optional but recommended step periodically checks if the camera has been moved, invalidating the calibration.

1.  **Schedule Check:** Run this check at a regular interval (e.g., once per hour).
2.  **Get Current Frame:** Capture a new frame from the camera and convert it to grayscale.
3.  **Calculate Gradient:** Compute a gradient image using the `Sobel` or `Scharr` filter. This image highlights edges.
4.  **Verify Edges:**
    * For each of the four line segments that define the **stored** island boundary:
    * Sample a few points along the line's path.
    * For each sample point, look up the pixel value in the **gradient image**.
    * Sum the gradient values along each of the four edges.
5.  **Check Threshold:** If the camera has not moved, the summed gradient values will be high (as they fall on the island's edge). If the camera has been bumped, the expected edge location is now on a flat surface, and the gradient sum will be very low.
6.  **Trigger Alert:** If the sum for any edge drops below a pre-defined threshold, flag the system that re-calibration is required.