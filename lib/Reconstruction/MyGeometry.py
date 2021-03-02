import numpy as np
import odl
def elekta_icon_geometry(sad=780.0, sdd=1000.0,
                         piercing_point=(390.0, 0.0),
                         angles=None, num_angles=None,
                         detector_shape=(780, 720)):
    sad = float(sad)
    assert sad > 0
    sdd = float(sdd)
    assert sdd > sad
    piercing_point = np.array(piercing_point, dtype=float)
    assert piercing_point.shape == (2,)
    if angles is not None and num_angles is not None:
        raise ValueError('cannot provide both `angles` and `num_angles`')
    elif angles is not None:
        angles = odl.nonuniform_partition(angles)
        assert angles.ndim == 1
    elif num_angles is not None:
        angles = odl.uniform_partition(1.2, 5.0, num_angles)
    else:
        angles = odl.uniform_partition(1.2, 5.0, 332)
    detector_shape = np.array(detector_shape, dtype=int)
    # Constant system parameters
    pixel_size = 0.368
    det_extent_mm = np.array([287.04, 264.96])
    # Compute the detector partition
    piercing_point_mm = pixel_size * piercing_point
    det_min_pt = -piercing_point_mm
    det_max_pt = det_min_pt + det_extent_mm
    detector_partition = odl.uniform_partition(min_pt=det_min_pt,
                                               max_pt=det_max_pt,
                                               shape=detector_shape)
    # Create the geometry
    geometry = odl.tomo.ConeFlatGeometry(
        angles, detector_partition,
        src_radius=sad, det_radius=sdd - sad)
    return geometry
