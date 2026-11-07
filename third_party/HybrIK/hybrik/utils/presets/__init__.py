from .simple_transform import SimpleTransform
from .simple_transform_3d_smpl import SimpleTransform3DSMPL
from .simple_transform_3d_smpl_cam import SimpleTransform3DSMPLCam
from .simple_transform_cam import SimpleTransformCam
# [copyright_detection 수정] SMPL-X 변환기는 import 시 SMPL-X 모델 파일 4개를 읽는다.
# 이 프로젝트의 motion 은 SMPL 경로(SimpleTransform3DSMPLCam)만 쓰므로, 파일이 없으면 건너뛴다.
try:
    from .simple_transform_3d_smplx import SimpleTransform3DSMPLX
except (AssertionError, FileNotFoundError, OSError):
    SimpleTransform3DSMPLX = None

__all__ = [
    'SimpleTransform', 'SimpleTransform3DSMPL', 'SimpleTransform3DSMPLCam', 'SimpleTransformCam',
    'SimpleTransform3DSMPLX']
