# simulated-av-lane-assist
A simulated autonomous driving project implementing a lane-keeping assistant using a hybrid Python-C++ architecture with MLOps principles.

## Project Structure

```
├── .gitignore
├── README.md
├── app/
├── data/
├── docs/
├── models/
└── src/
    ├── cpp/
    └── python/
```
### Dataset Generation - Run Configurations

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
