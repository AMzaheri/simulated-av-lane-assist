# Simulated Driving Dataset Details

## Purpose of Data
 
This dataset was generated to provide diverse and challenging input for training end-to-end steering models for simulated autonomous vehicles.

## Dataset Generation - Run Configurations


This project uses `src/python/data_generator.py` to simulate car driving and collect image-label pairs. Different configurations are used to generate varied datasets, stored in versioned subdirectories under the `data/` folder.

#### Run `run_v1_CarClose2Centre`

**Purpose:** To generate a dataset focused on keeping the car close to the lane center, ensuring both lane lines are always visible. This configuration aims for stable driving behavior.

**Key Parameters used in `data_generator.py` for this run:**

* **`CURRENT_RUN_NAME`**: `run_v1_CarClose2Centre`
* **`NUM_SAMPLES`**: 6400 (This run is optimized to maintain stable capture of both lines up to this number of images. For significantly larger datasets, further tuning might be needed.)
* **Random Steering (Frequency)**: `np.random.rand() < 0.03` (3% chance per frame for random steering input).
* **Random Steering (Magnitude)**: `np.random.uniform(0.1, 0.2)` (Small, gentle steering nudges).
* **Safe Bounds (`safe_left_bound`, `safe_right_bound`)**: No pixel buffer (car corrects as soon as its edge touches the road boundary).
* **Boundary Correction Strength (`car.steer(X)` when near X-bounds)**: `steer(-4)` for left bound, `steer(4)` for right bound (Strong correction to keep car centered).
* **Angle Correction Thresholds (`car.angle > X` / `< Y`)**: `95` degrees (for left angle), `85` degrees (for right angle) (Correction kicks in sooner for small deviations).
* **Angle Correction Strength (`car.steer(X)` when angle deviates)**: `steer(-2)` for angle > 95, `steer(2)` for angle < 85 (Moderate correction for angle deviations).
* **Car X-Position Reset on Loop**: `SCREEN_WIDTH / 2 + np.random.uniform(-20, 20)` (Car's X-position resets to center +/- 20 pixels when looping).

This configuration yields a dataset of consistently centered driving, ideal for initial model training where stable lane-following is the primary focus.

#### Run `run_v2_Diversified_WiderRangeOffsets`

**Purpose:** To generate a more diverse dataset by allowing the car to explore a wider range of horizontal offsets within the lane. This helps the model learn to handle situations where the car is not perfectly centered.

**Key Parameters used in `data_generator.py` for this run:**

* **`CURRENT_RUN_NAME`**: `run_v2_Diversified_WiderRangeOffsets`
* **`NUM_SAMPLES`**: 5000 (Generated after stabilization from `run_v1`, focusing on horizontal diversity.)
* **Random Steering (Frequency)**: `np.random.rand() < 0.15` (Increased from 0.03; 15% chance per frame for random steering input).
* **Random Steering (Magnitude)**: `np.random.uniform(0.1, 0.4)` (Increased from 0.1-0.2; allowing slightly larger steering nudges).
* **Safe Bounds (`safe_left_bound`, `safe_right_bound`)**: No pixel buffer (car corrects as soon as its edge touches the road boundary) - *Same as `run_v1`*.
* **Boundary Correction Strength (`car.steer(X)` when near X-bounds)**: `steer(-4)` for left bound, `steer(4)` for right bound (Strong correction) - *Same as `run_v1`*.
* **Angle Correction Thresholds (`car.angle > X` / `< Y`)**: `95` degrees (for left angle), `85` degrees (for right angle) - *Same as `run_v1`*.
* **Angle Correction Strength (`car.steer(X)` when angle deviates)**: `steer(-2)` for angle > 95, `steer(2)` for angle < 85 - *Same as `run_v1`*.
* **Car X-Position Reset on Loop**: `SCREEN_WIDTH / 2 + np.random.uniform(-20, 20)` - *Same as `run_v1`*.
* **No Empty Road Frames**: Includes logic to prevent saving images where the car is off-screen during loop reset.

This dataset provides more varied horizontal car positions while maintaining line visibility, offering richer training data for robust lane-following.

#### Run `run_v3_Diversified_RandomCARSPEED`

**Purpose:** To enhance dataset diversity by introducing variations in the car's speed during simulation. This helps the model become more robust to different driving speeds in real-world scenarios.

**Key Parameters used in `data_generator.py` for this run:**

* **`CURRENT_RUN_NAME`**: `run_v3_Diversified_RandomCARSPEED`
* **`NUM_SAMPLES`**: 5000
* **Car Speed Variation**: `np.random.uniform(CAR_SPEED * 0.8, CAR_SPEED * 1.2)` (Car speed randomly varies between 80% and 120% of the base `CAR_SPEED`).
* **Random Steering (Frequency)**: `np.random.rand() < 0.15` (15% chance).
* **Random Steering (Magnitude)**: `np.random.uniform(0.1, 0.4)`.
* **Other parameters**: Boundary correction strength (`steer(4)`), angle correction (`steer(2)` with `95/85` thresholds), and X-position reset on loop (`+/- 20px`) remain consistent with `run_v2`.

This dataset adds speed variation to the horizontally diversified data.

#### Run `run_v4_VariedCameraPosition`

**Purpose:** To further diversify the dataset by simulating variations in the camera's vertical mounting position relative to the car. This improves the model's robustness to slight changes in camera setup or vehicle pitch/roll.

**Key Parameters used in `data_generator.py` for this run:**

* **`CURRENT_RUN_NAME`**: `run_v4_VariedCameraPosition`
* **`NUM_SAMPLES`**: 5000
* **Camera Position Variation**: `car.camera_offset_y = base_camera_offset_y + np.random.uniform(-10, 10)` (Camera's Y-offset from car center varies randomly by +/- 10 pixels from its base value).
* **Car Speed Variation**: `np.random.uniform(CAR_SPEED * 0.8, CAR_SPEED * 1.2)`.
* **Random Steering (Frequency)**: `np.random.rand() < 0.15`.
* **Random Steering (Magnitude)**: `np.random.uniform(0.1, 0.4)`.
* **Other parameters**: Boundary correction strength (`steer(4)`), angle correction (`steer(2)` with `95/85` thresholds), X-position reset on loop (`+/- 20px`), and logic to prevent empty road frames remain consistent.

This dataset adds camera perspective variability, making the training data more comprehensive and applicable to a wider range of real-world conditions.

#### `run_v5_CurveMovement`

* **Purpose:** This run focuses on generating data for a **curved road scenario** specifically for a **left-hand turn**. It incorporates a refined automated driving logic to ensure the car follows the curve accurately and consistently.
* **Road Type:** Curved (Left Turn: from 90 to 180 degrees relative to the curve center)
* **Key Features/Improvements:**
    * **Corrected Curved Road Following:** Implemented a pure pursuit-like controller to accurately guide the car along the defined arc, addressing previous issues with incorrect turning direction and off-road movement.
    * **Arc-Specific Reset Logic:** The car now automatically resets to the beginning of the curved path (top of the arc) once it successfully completes the defined arc (reaches the 180-degree mark) or if it veers significantly off track. This ensures that all generated images contain relevant road data.
    * **Automated Data Collection:** Configured to collect a large number of samples (e.g., 5000 images) suitable for training a machine learning model.
* **Data Structure:**
    * Images are saved in the `data/run_v5_CurveMovement/images/` subdirectory.
    * Corresponding steering angles are recorded in `data/run_v5_CurveMovement/labels.csv`.
* **Considerations:**
    * The generated images contain black areas due to the rectangular camera view on a curved road. This is intentional and mimics real-world scenarios, promoting a more robust model that learns to focus on the road features.
    * The `KP_ANGLE` and `KP_OFFSET` parameters were tuned for stable curve following. Further tuning might be required based on model performance.


#### `run_v6_DiversifiedCurvedMovement`

* **Purpose:** This run builds upon the previous curved road data generation by introducing significant diversification to enhance the robustness and generalisation capabilities of the trained machine learning model.
* **Road Type:** Curved (Left Turn: from 90 to 180 degrees relative to the curve center)
* **Key Features/Improvements:**
    * **Dynamic Target Lateral Offset:**
        * Introduced a `target_lateral_offset` variable that periodically changes, making the car intentionally drive slightly off the lane center.
        * The `steering_label` calculation now incorporates this offset (`effective_offset_error`), training the model to handle and correct for varying lateral positions within the lane.
        * Parameters like `offset_change_timer` and `OFFSET_CHANGE_INTERVAL` control the frequency of these target offset changes.
    * **Randomised Reset Position and Angle:**
        * When the car resets to the start of the curved path, its initial `x` position and `angle` are now randomly perturbed using `reset_lateral_offset` and `reset_angle_deviation`.
        * This ensures the car starts each segment from slightly different initial conditions, forcing the controller (and model) to adapt and correct immediately.
    * **Automated Data Collection:** Configured to collect a large number of samples (e.g., 5000 images) suitable for training a machine learning model.
* **Data Structure:**
    * Images are saved in the `data/run_v6_DiversifiedCurvedMovement/images/` subdirectory.
    * Corresponding steering angles are recorded in `data/run_v6_DiversifiedCurvedMovement/labels.csv`.
* **Considerations:**
    * The generated images continue to contain some black areas outside the road and minor rendering artifacts on the road surface. These are considered beneficial for model robustness, mimicking real-world visual variations.
    * The `KP_ANGLE` and `KP_OFFSET` parameters, along with the new diversification parameters, can be further tuned based on model performance.

#### Run `run_v7_DiversifiedCurvedMovement_v01`

 * Most parameters are similiar to  as run `run_v7_DiversifiedCurvedMovement`. Adjusted `target_lateral_offset` range (in `generate_data`):

` * target_lateral_offset = np.random.uniform(-LANE_WIDTH / 2, LANE_WIDTH / 2)`. I increased the `LANE_WIDTH / 3` value to make the car wander less aggressively from the lane center. A larger range means more diverse lateral positions.


#### Run `run_v7_DiversifiedCurvedMovement_v02`

 * Similar to `run_v7_DiversifiedCurvedMovement_v01` with `target_lateral_offset = np.random.uniform(-LANE_WIDTH, LANE_WIDTH)`

 
#### Run `run_v7_DiversifiedCurvedMovement_v03`

 * `OFFSET_CHANGE_INTERVAL` set to `FPS * 1`. Other parameters are similar to `run_v7_DiversifiedCurvedMovement_v02`.
 * FPS * 3: This controls how frequently the target_lateral_offset changes. Decreasing this value (e.g., FPS * 1 for every 1 second) will make the car's lateral target change more often, leading to more frequent corrections and greater diversity in the car's path. Increasing it will make the car follow a specific offset for longer.
 

#### Run `run_v7_DiversifiedCurvedMovement_v04`

 * Changed `reset_lateral_offset` range to `np.random.uniform(-LANE_WIDTH / 2.25, LANE_WIDTH / 2.25)`. This parameter controls how far off-center the car starts after a reset. A larger range makes the initial correction task more challenging.
 * Other parameters similar to thoes in `run_v7_DiversifiedCurvedMovement_v03`

#### Run `run_v7_DiversifiedCurvedMovement_v05`

 * Changed `reset_lateral_offset` range to `np.random.uniform(-LANE_WIDTH / 2.05, LANE_WIDTH / 2.05)`.
 * Other parameters similar to thoes in `run_v7_DiversifiedCurvedMovement_v03`.

#### Run `run_v7_DiversifiedCurvedMovement_v06 to run_v7_DiversifiedCurvedMovement_v08`

 * Adjusted `reset_angle_deviation` range to
  * ` = np.random.uniform(-4, 4)`
  * ` = np.random.uniform(-10, 10)`
  * ` = np.random.uniform(-20, 20)`
 * This parameter controls how much the car's starting angle deviates from the perfect tangent after a reset. A larger range forces the model to handle more significant initial yaw errors.
 * Other parameters: similar to `run_v7_DiversifiedCurvedMovement_v05`
 
#### Run `run_v7_DiversifiedCurvedMovement_v09`

 * Widened car speed variation `min_speed = CAR_SPEED * 0.7, max_speed = CAR_SPEED * 1.3`
 * Other parameters: similar to `run_v7_DiversifiedCurvedMovement_v08`


#### Run `run_v7_DiversifiedCurvedMovement_v10`

 * Adjusted random steering component.
  * Used `if np.random.rand() < 0.05`
  * `steering_label += np.random.uniform(-0.6, 0.6)`: A larger range will create more unpredictable, human-like driving data.


  #### Run `run_v7_DiversifiedCurvedMovement_v11`

 * Adjusted random rteering component:the probability (0.05 for 10%) of this random steering component being applied.
  * `if np.random.rand() < 0.1:`
  * `steering_label += np.random.uniform(-0.6, 0.6)`
 * Other parameters: similar to `run_v7_DiversifiedCurvedMovement_v10`


#### Run `run_v7_DiversifiedCurvedMovement_v12 and run_v7_DiversifiedCurvedMovement_v13`

 * Adjusted Camera Perspective:camera vertical offset to
  * `car.camera_offset_y = base_camera_offset_y + np.random.uniform(-30, 30)`
  * `car.camera_offset_y = base_camera_offset_y + np.random.uniform(-100, 100)`
  * This value controls how high or low the camera is positioned relative to the car's center. This simulates different vehicle suspensions or camera mounting points.
 * Other parameters: similar to `run_v7_DiversifiedCurvedMovement_v11`  

#### Run `run_v7_DiversifiedCurvedMovement_v14`

 * Chnaged control parameter, `LOOK_AHEAD_DISTANCE` to 120
 * Note: Changing this can alter the "aggressiveness" of the car's turn. A larger look-ahead makes turns smoother but might result in wider turns; a smaller one makes turns sharper.
 * Other parameter similar to `run_v7_DiversifiedCurvedMovement_v12` 

#### Run `run_v7_DiversifiedCurvedMovement_v15'

 * Tuned another contol parameter (`KP_ANGLE`)
 * `KP_ANGLE and KP_OFFSET: (0.6 and 0.05 respectively)`
 * Note: These proportional gains determine how strongly the car corrects for angle and offset errors. Varying these slightly can produce data where the car corrects more slowly/quickly, leading to different steering angle distributions. Be careful, as large changes here can make the controller unstable and cause the car to drive off the road.
 * Other parameters: similar to `run_v7_DiversifiedCurvedMovement_v12`
