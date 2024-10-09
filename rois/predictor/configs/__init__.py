from .sort import sort


"""
PREDICTOR_MODELS

A dictionary that maps predictor names to their respective tracking configurations.

Keys:
    str: Predictor names (e.g., "sort").

Values:
    type: The corresponding tracking configuration, which includes:
        - sort: Configuration for the SORT (Simple Online and Realtime Tracking) tracker.

Usage:
    Access the tracker configuration by referring to its corresponding predictor name. For example:
        tracker_config = PREDICTOR_MODELS["sort"]
    
    You can add custom trackers by defining them in the respective module files, then use them for tracking in your applications. For example:
        CustomTracker = dict(
            module_name='my_custom_tracker_module',
            class_name='MyCustomTracker',
            args=dict(
                custom_int=6,
            ),
            frame_delay=2,
        )
        
        PREDICTOR_MODELS = {
            "sort": sort,
            "custom": CustomTracker,
        }
"""
PREDICTOR_MODELS = {
    "sort": sort,
}
