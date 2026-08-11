import os

_SKIP_AUTOREGISTER = os.environ.get('MOTIUS_SKIP_AUTOREGISTER', '').lower() in {
    '1', 'true', 'yes',
}
if _SKIP_AUTOREGISTER:
    # Evaluation scripts import the dependency-light universal mask builder
    # directly. Do not eagerly import unrelated audio, VerMo, or SMPL modules
    # when global registry side effects were explicitly disabled.
    from motius.datasets.motion.motionhub.transforms.universal_mask import (
        PrepareM2MUniversalMask,
    )

    __all__ = ['PrepareM2MUniversalMask']
else:
    from motius.datasets.motion.motionhub.transforms.compose_multi_person import ComposeMultiPerson
    from motius.datasets.motion.motionhub.transforms.crop import (
        CropMotionByTextTime,
        MotionAudioMaxDurationFilter,
        MotionAudioRandomCrop,
        RandomCropPadding,
    )
    from motius.datasets.motion.motionhub.transforms.formatting import PackInputs, ToTensor
    from motius.datasets.motion.motionhub.transforms.load_audio import LoadAudio
    from motius.datasets.motion.motionhub.transforms.load_o6dp import LoadO6dp
    from motius.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
    from motius.datasets.motion.motionhub.transforms.load_text import (
        LoadCompatibleCaption,
        LoadHierarchicalCaption,
        LoadHm3dTxt,
        LoadHYMotionCaption,
        LoadTxt,
    )
    from motius.datasets.motion.motionhub.transforms.remap_path import RemapMotionPathToO6dp
    from motius.datasets.motion.motionhub.transforms.split_for_ar import SplitMotionForAR, SplitMusicForAR
    from motius.datasets.motion.motionhub.transforms.split_motion import (
        PrepareM2MCompletion,
        SplitInbetween,
        SplitPrediction,
    )
    from motius.datasets.motion.motionhub.transforms.local_to_global import (
        LocalToGlobalRotation,
    )
    from motius.datasets.motion.motionhub.transforms.universal_mask import (
        PrepareM2MUniversalMask,
    )
    from motius.datasets.motion.motionhub.transforms.crop_audio_to_motion import CropAudioToMotion
    from motius.datasets.motion.motionhub.transforms.compute_198dim import (
        Compute198DimPosition,
        Compute201DimO6DP,
    )
    from motius.datasets.motion.motionhub.transforms.prepare_m2m import PrepareM2MCondition
    from motius.datasets.motion.motionhub.transforms.prepare_m2m_v2_fullmask import (
        PrepareM2Mv2FullMask,
    )
    from motius.datasets.motion.motionhub.transforms.prepare_m2m_v2_overfit_case import (
        PrepareM2Mv2OverfitCase,
    )
    from motius.datasets.motion.motionhub.transforms.smpl_trans_to_kimodo_root import (
        SmplTransToKimodoRootOnline,
    )
    from motius.datasets.motion.motionhub.transforms.load_editing_source import (
        LoadEditingSourceMotion,
    )
    from motius.datasets.motion.motionhub.transforms.compute_147dim import (
        Compute147DimEndEffector,
    )
    from motius.datasets.motion.motionhub.transforms.compute_151dim_contact import (
        Compute151DimFootContact,
    )
    from motius.datasets.motion.motionhub.transforms.dimension_filter import (
        EnsureDimensionFilter,
    )

    __all__ = [
        'ComposeMultiPerson',
        'CropMotionByTextTime',
        'MotionAudioMaxDurationFilter',
        'MotionAudioRandomCrop',
        'RandomCropPadding',
        'PackInputs',
        'ToTensor',
        'LoadAudio',
        'LoadO6dp',
        'LoadSmplx55',
        'LoadCompatibleCaption',
        'LoadHierarchicalCaption',
        'LoadHm3dTxt',
        'LoadHYMotionCaption',
        'LoadTxt',
        'SplitMotionForAR',
        'SplitMusicForAR',
        'PrepareM2MCompletion',
        'PrepareM2MUniversalMask',
        'LocalToGlobalRotation',
        'Compute198DimPosition',
        'Compute201DimO6DP',
        'PrepareM2MCondition',
        'PrepareM2Mv2FullMask',
        'PrepareM2Mv2OverfitCase',
        'CropAudioToMotion',
        'RemapMotionPathToO6dp',
        'SplitInbetween',
        'SplitPrediction',
        'SmplTransToKimodoRootOnline',
        'LoadEditingSourceMotion',
        'Compute147DimEndEffector',
        'Compute151DimFootContact',
        'EnsureDimensionFilter',
    ]
