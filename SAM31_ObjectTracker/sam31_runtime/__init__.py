"""SAM 3.1 ArcGIS ObjectTracker runtime helpers.

Modules:
  model_loader   - build the SAM 3.1 multiplex tracker once from package-relative assets
  frame_adapter  - ArcGIS frame ndarray -> SAM 3.1 input tensor
  arcgis_boxes   - ArcGIS box parsing / mask -> box conversion
  sam31_session  - persistent streaming tracking session (Object Multiplex state)
  logging_utils  - restrained logging
"""

__version__ = "0.2.0"
