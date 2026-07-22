"""D2M-GAN's public 86-segment AIST++ dance-to-music test split."""

from __future__ import annotations

from dataclasses import dataclass


# Source: https://github.com/L-YeZhu/D2M-GAN/blob/main/dataset/
# aist_motion_test_segment.txt
_TEST_IDS = (
    "gPO_sBM_c01_d11_mPO1_ch02_seg1",
    "gLH_sBM_c01_d17_mLH4_ch02_seg2",
    "gLO_sBM_c01_d15_mLO2_ch02_seg1",
    "gMH_sBM_c01_d24_mMH3_ch02_seg3",
    "gJS_sBM_c01_d01_mJS3_ch02_seg3",
    "gJB_sBM_c01_d09_mJB5_ch02_seg3",
    "gHO_sBM_c01_d20_mHO5_ch02_seg3",
    "gLH_sBM_c01_d17_mLH4_ch02_seg4",
    "gBR_sBM_c01_d04_mBR0_ch02_seg4",
    "gWA_sBM_c01_d26_mWA0_ch02_seg5",
    "gMH_sBM_c01_d24_mMH3_ch02_seg1",
    "gKR_sBM_c01_d28_mKR2_ch02_seg3",
    "gBR_sBM_c01_d05_mBR0_ch02_seg1",
    "gLH_sBM_c01_d17_mLH4_ch02_seg1",
    "gPO_sBM_c01_d10_mPO1_ch02_seg1",
    "gMH_sBM_c01_d24_mMH3_ch02_seg2",
    "gPO_sBM_c01_d11_mPO1_ch02_seg2",
    "gBR_sBM_c01_d04_mBR0_ch02_seg1",
    "gKR_sBM_c01_d28_mKR2_ch02_seg1",
    "gKR_sBM_c01_d30_mKR2_ch02_seg1",
    "gLO_sBM_c01_d15_mLO2_ch02_seg2",
    "gPO_sBM_c01_d10_mPO1_ch02_seg3",
    "gWA_sBM_c01_d26_mWA0_ch02_seg3",
    "gJS_sBM_c01_d03_mJS3_ch02_seg4",
    "gLO_sBM_c01_d13_mLO2_ch02_seg4",
    "gHO_sBM_c01_d21_mHO5_ch02_seg2",
    "gLO_sBM_c01_d15_mLO2_ch02_seg3",
    "gWA_sBM_c01_d26_mWA0_ch02_seg2",
    "gMH_sBM_c01_d24_mMH3_ch02_seg4",
    "gLO_sBM_c01_d13_mLO2_ch02_seg1",
    "gPO_sBM_c01_d11_mPO1_ch02_seg3",
    "gWA_sBM_c01_d25_mWA0_ch02_seg5",
    "gJS_sBM_c01_d03_mJS3_ch02_seg2",
    "gBR_sBM_c01_d05_mBR0_ch02_seg4",
    "gLH_sBM_c01_d17_mLH4_ch02_seg3",
    "gBR_sBM_c01_d04_mBR0_ch02_seg3",
    "gKR_sBM_c01_d30_mKR2_ch02_seg4",
    "gMH_sBM_c01_d22_mMH3_ch02_seg3",
    "gLH_sBM_c01_d18_mLH4_ch02_seg4",
    "gHO_sBM_c01_d21_mHO5_ch02_seg1",
    "gMH_sBM_c01_d22_mMH3_ch02_seg4",
    "gWA_sBM_c01_d26_mWA0_ch02_seg1",
    "gLO_sBM_c01_d13_mLO2_ch02_seg2",
    "gWA_sBM_c01_d25_mWA0_ch02_seg4",
    "gKR_sBM_c01_d28_mKR2_ch02_seg2",
    "gPO_sBM_c01_d11_mPO1_ch02_seg4",
    "gBR_sBM_c01_d04_mBR0_ch02_seg2",
    "gJB_sBM_c01_d08_mJB5_ch02_seg1",
    "gKR_sBM_c01_d30_mKR2_ch02_seg2",
    "gLH_sBM_c01_d18_mLH4_ch02_seg1",
    "gBR_sBM_c01_d04_mBR0_ch02_seg6",
    "gJS_sBM_c01_d01_mJS3_ch02_seg1",
    "gBR_sBM_c01_d05_mBR0_ch02_seg2",
    "gJS_sBM_c01_d01_mJS3_ch02_seg4",
    "gPO_sBM_c01_d10_mPO1_ch02_seg5",
    "gKR_sBM_c01_d28_mKR2_ch02_seg4",
    "gBR_sBM_c01_d05_mBR0_ch02_seg6",
    "gMH_sBM_c01_d22_mMH3_ch02_seg2",
    "gLO_sBM_c01_d13_mLO2_ch02_seg3",
    "gKR_sBM_c01_d30_mKR2_ch02_seg3",
    "gJB_sBM_c01_d08_mJB5_ch02_seg2",
    "gMH_sBM_c01_d22_mMH3_ch02_seg1",
    "gBR_sBM_c01_d04_mBR0_ch02_seg5",
    "gLO_sBM_c01_d15_mLO2_ch02_seg4",
    "gWA_sBM_c01_d26_mWA0_ch02_seg4",
    "gJB_sBM_c01_d08_mJB5_ch02_seg3",
    "gBR_sBM_c01_d05_mBR0_ch02_seg3",
    "gJB_sBM_c01_d09_mJB5_ch02_seg1",
    "gHO_sBM_c01_d21_mHO5_ch02_seg3",
    "gPO_sBM_c01_d10_mPO1_ch02_seg4",
    "gWA_sBM_c01_d25_mWA0_ch02_seg1",
    "gPO_sBM_c01_d10_mPO1_ch02_seg2",
    "gLH_sBM_c01_d18_mLH4_ch02_seg2",
    "gJS_sBM_c01_d03_mJS3_ch02_seg1",
    "gWA_sBM_c01_d25_mWA0_ch02_seg3",
    "gBR_sBM_c01_d05_mBR0_ch02_seg5",
    "gHO_sBM_c01_d20_mHO5_ch02_seg2",
    "gJS_sBM_c01_d03_mJS3_ch02_seg3",
    "gWA_sBM_c01_d25_mWA0_ch02_seg6",
    "gLH_sBM_c01_d18_mLH4_ch02_seg3",
    "gHO_sBM_c01_d20_mHO5_ch02_seg1",
    "gJS_sBM_c01_d01_mJS3_ch02_seg2",
    "gPO_sBM_c01_d11_mPO1_ch02_seg5",
    "gWA_sBM_c01_d26_mWA0_ch02_seg6",
    "gWA_sBM_c01_d25_mWA0_ch02_seg2",
    "gJB_sBM_c01_d09_mJB5_ch02_seg2",
)


@dataclass(frozen=True)
class D2MGANAISTPPSegment:
    case_id: str
    source_motion_id: str
    segment_index: int
    music_id: str
    start_seconds: float
    duration_seconds: float = 2.0


def _parse(case_id: str) -> D2MGANAISTPPSegment:
    fields = case_id.split("_")
    if len(fields) != 7 or not fields[-1].startswith("seg"):
        raise ValueError(f"Invalid D2M-GAN AIST++ segment id: {case_id!r}")
    segment_index = int(fields[-1][3:])
    source_motion_id = "_".join((fields[0], fields[1], "cAll", *fields[3:6]))
    return D2MGANAISTPPSegment(
        case_id=case_id,
        source_motion_id=source_motion_id,
        segment_index=segment_index,
        music_id=fields[4],
        start_seconds=(segment_index - 1) * 2.0,
    )


def d2mgan_aistpp_test_segments() -> tuple[D2MGANAISTPPSegment, ...]:
    return tuple(_parse(case_id) for case_id in _TEST_IDS)


__all__ = ["D2MGANAISTPPSegment", "d2mgan_aistpp_test_segments"]
