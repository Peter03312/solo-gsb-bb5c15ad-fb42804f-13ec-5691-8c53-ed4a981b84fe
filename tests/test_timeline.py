"""时间线合并与预算算法单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.timeline import (
    DEVIATION_OVER,
    DEVIATION_UNDER,
    DeviationAnomaly,
    Segment,
    attribute_exposure,
    clip_segments,
    coverage_gaps,
    first_exceed_time,
    integrate,
    integrate_within,
    merge_intervals,
    reconcile_illuminance,
    segments_piecewise_equal,
    simplify_intervals,
    year_bounds,
)


def _dt(h: float, day: int = 1) -> datetime:
    return datetime(2026, 3, day, tzinfo=timezone.utc) + timedelta(hours=h)


def test_disjoint_intervals_kept_separate():
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(4), _dt(6), 100)])
    assert [(s.start, s.end, s.lux) for s in segs] == [
        (_dt(0), _dt(2), 50),
        (_dt(4), _dt(6), 100),
    ]


def test_overlap_takes_max_lux_not_sum():
    # 两盏灯在 10:00-12:00 重叠：有效照度取 max(50, 200)=200
    segs = merge_intervals([(_dt(8), _dt(12), 50), (_dt(10), _dt(14), 200)])
    assert segs == [
        Segment(_dt(8), _dt(10), 50),
        Segment(_dt(10), _dt(14), 200),
    ]
    assert integrate(segs) == 50 * 2 + 200 * 4


def test_triple_overlap_and_coalescing():
    segs = merge_intervals(
        [(_dt(0), _dt(3), 50), (_dt(2), _dt(5), 80), (_dt(4), _dt(6), 80)]
    )
    # 0-2:50, 2-3:max(50,80)=80, 3-4:80, 4-5:max(80,80)=80, 5-6:80
    # 相邻同照度合并：2-6 连成一段 80
    assert segs == [Segment(_dt(0), _dt(2), 50), Segment(_dt(2), _dt(6), 80)]


def test_touching_intervals_are_not_overlap():
    # 首尾相接不算重叠，但同照度会被合并为一个片段
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(2), _dt(4), 50)])
    assert segs == [Segment(_dt(0), _dt(4), 50)]


def test_zero_effective_lux_dropped_when_brighter_lamp_active_then_ends():
    # 0 lux 区间与有灯区间重叠时不产生影响
    segs = merge_intervals([(_dt(0), _dt(4), 0), (_dt(1), _dt(3), 50)])
    assert segs == [Segment(_dt(1), _dt(3), 50)]


def test_clip_to_year_window():
    start, end = year_bounds(2026)
    segs = merge_intervals(
        [
            (datetime(2025, 12, 31, 20, tzinfo=timezone.utc),
             datetime(2026, 1, 1, 4, tzinfo=timezone.utc), 100)
        ]
    )
    clipped = clip_segments(segs, start, end)
    assert clipped == [
        Segment(start, datetime(2026, 1, 1, 4, tzinfo=timezone.utc), 100)
    ]
    assert integrate_within(segs, start, end) == 400


def test_first_exceed_time_interpolated():
    # 恒定 100 lux，上限 250 lux·h → 2.5 小时处超标
    segs = merge_intervals([(_dt(0), _dt(10), 100)])
    exceed = first_exceed_time(segs, 250.0, year_bounds(2026)[0])
    assert exceed == _dt(2.5)


def test_exactly_at_limit_is_not_exceeded():
    segs = merge_intervals([(_dt(0), _dt(5), 50)])  # 恰好 250
    assert first_exceed_time(segs, 250.0, year_bounds(2026)[0]) is None


def test_first_exceed_skips_low_exposure_segment():
    # 第一段 2h*50=100，第二段从 2h 起 200lux，还需 150 → 0.75h
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(2), _dt(6), 200)])
    exceed = first_exceed_time(segs, 250.0, year_bounds(2026)[0])
    assert exceed == _dt(2.75)


def test_coverage_gaps_merges_dark_and_dim_periods():
    # 08-12 @50 达标；12-14 无灯、14-16 @30 不达标、16-18 无灯 → 连成单一缺口
    segs = merge_intervals([(_dt(8), _dt(12), 50), (_dt(14), _dt(16), 30)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == [(_dt(12), _dt(18))]
    assert compliant == 4.0


def test_coverage_gaps_exact_threshold_is_compliant():
    # 照度恰好等于门槛 → 达标，无缺口
    segs = merge_intervals([(_dt(8), _dt(18), 50)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == []
    assert compliant == 10.0


def test_coverage_gaps_no_light_at_all():
    # 完全无灯：整个窗口是一个缺口
    gaps, compliant = coverage_gaps([], _dt(8), _dt(18), 50)
    assert gaps == [(_dt(8), _dt(18))]
    assert compliant == 0.0


def test_coverage_gaps_split_around_bright_period():
    # 10-12 @100 达标，前后各一段缺口
    segs = merge_intervals([(_dt(10), _dt(12), 100)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == [(_dt(8), _dt(10)), (_dt(12), _dt(18))]
    assert compliant == 2.0


def test_coverage_gaps_clips_cross_day_segment_to_window():
    # 跨日片段 20:00 → 次日 10:00：当日窗口 08:00-12:00 无交集，次日窗口取 08:00-10:00
    segs = merge_intervals([(_dt(20), _dt(10, day=2), 80)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(12), 50)
    assert gaps == [(_dt(8), _dt(12))]
    assert compliant == 0.0
    gaps, compliant = coverage_gaps(segs, _dt(8, day=2), _dt(12, day=2), 50)
    assert gaps == [(_dt(10, day=2), _dt(12, day=2))]
    assert compliant == 2.0


# ------------------------------------------------------------- 曝光归因

def test_attribute_exposure_overlap_takes_max_and_shadows():
    # A: 08-18 @50；B: 10-12 @200 → 重叠段取 max，A 被遮蔽 2 小时
    total, attrs = attribute_exposure(
        [(1, _dt(8), _dt(18), 50), (2, _dt(10), _dt(12), 200)], _dt(8), _dt(18)
    )
    assert total == 50 * 2 + 200 * 2 + 50 * 6
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 50 * 2 + 50 * 6
    assert by_id[1].shadowed_hours == 2.0
    assert by_id[2].lux_hours == 200 * 2
    assert by_id[2].shadowed_hours == 0.0


def test_attribute_exposure_tied_brightest_split_evenly():
    # 两条 100 lux 并列最亮：该片的 200 lux·h 平分；50 lux 不获归因只记遮蔽
    total, attrs = attribute_exposure(
        [
            (1, _dt(10), _dt(12), 100),
            (2, _dt(10), _dt(12), 100),
            (3, _dt(10), _dt(12), 50),
        ],
        _dt(8),
        _dt(18),
    )
    assert total == 200.0
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 100.0
    assert by_id[2].lux_hours == 100.0
    assert by_id[3].lux_hours == 0.0
    assert by_id[3].shadowed_hours == 2.0


def test_attribute_exposure_clips_cross_boundary_intervals():
    # 跨界区间只计入窗口内部分；窗口外部分不产生归因与遮蔽
    total, attrs = attribute_exposure(
        [(1, _dt(20), _dt(10, day=2), 80), (2, _dt(21), _dt(23), 30)],
        _dt(22),
        _dt(23),
    )
    assert total == 80.0
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 80.0
    assert by_id[1].shadowed_hours == 0.0
    assert by_id[2].lux_hours == 0.0
    assert by_id[2].shadowed_hours == 1.0


def test_attribute_exposure_zero_lux_and_empty_window():
    # 0 lux 区间相交但不获归因也不被遮蔽；窗口内无区间时总量为 0、列表为空
    total, attrs = attribute_exposure([(1, _dt(8), _dt(10), 0)], _dt(8), _dt(18))
    assert total == 0.0
    assert [a.interval_id for a in attrs] == [1]
    assert attrs[0].lux_hours == 0.0
    assert attrs[0].shadowed_hours == 0.0

    total, attrs = attribute_exposure([], _dt(8), _dt(18))
    assert total == 0.0
    assert attrs == []


# ------------------------------------------------------------- 照明计划精简

def _iv(interval_id, h0, h1, lux, day=1):
    return (interval_id, _dt(h0, day), _dt(h1, day), lux)


def test_simplify_empty_returns_empty():
    result = simplify_intervals([])
    assert result.removable_ids == ()
    assert result.retained_ids == ()
    assert result.original_segments == ()
    assert result.simplified_segments == ()


def test_simplify_no_redundant_intervals_all_retained():
    # 两盏不同时段的灯：缺任一都会在原覆盖范围里留下暗段 → 全部保留
    result = simplify_intervals([_iv(1, 8, 12, 50), _iv(2, 13, 15, 50)])
    assert result.removable_ids == ()
    assert result.retained_ids == (1, 2)


def test_simplify_equal_lux_ties_keep_smallest_id():
    # 等照度并列：同照度同区间的副本，贪心从大到小撤去，只留最小 ID
    result = simplify_intervals([
        _iv(1, 8, 12, 50), _iv(2, 8, 12, 50), _iv(3, 8, 12, 50),
    ])
    assert result.removable_ids == (2, 3)
    assert result.retained_ids == (1,)


def test_simplify_dimmer_overlapped_by_brighter_is_removable():
    # 区间 1 全程被更亮的区间 2 压制（50 vs 200），撤去后逐时刻照度不变
    result = simplify_intervals([_iv(1, 8, 18, 50), _iv(2, 8, 18, 200)])
    assert result.removable_ids == (1,)
    assert result.retained_ids == (2,)


def test_simplify_different_lux_partial_shadow_not_redundant():
    # A: 08-18 @50；B: 10-12 @200。撤去 A 则 12-18 变暗；撤去 B 则 10-12 变暗
    result = simplify_intervals([_iv(1, 8, 18, 50), _iv(2, 10, 12, 200)])
    assert result.removable_ids == ()
    assert result.retained_ids == (1, 2)
    assert result.original_segments == (
        Segment(_dt(8), _dt(10), 50),
        Segment(_dt(10), _dt(12), 200),
        Segment(_dt(12), _dt(18), 50),
    )


def test_simplify_equal_exposure_but_different_shape_is_not_redundant():
    # 等照度并列副本可撤，但不能只比曝光积分：
    # A 08-10 @100 与 B 10-12 @100 曝光相同却覆盖不同时刻，二者都必须保留
    result = simplify_intervals([_iv(1, 8, 10, 100), _iv(2, 10, 12, 100)])
    assert result.removable_ids == ()
    assert result.retained_ids == (1, 2)

    # 照度形状不同也不能因积分相同而撤：50*4h 与 100*2h 总量相等
    result = simplify_intervals([_iv(1, 8, 12, 50), _iv(2, 8, 10, 100)])
    assert result.removable_ids == ()
    assert result.retained_ids == (1, 2)


def test_simplify_contained_same_lux_removable_in_id_order():
    # 同照度包含：大 ID 的被包含区间可撤；被包含的小 ID 撤去后覆盖不足，须保留
    result = simplify_intervals([_iv(1, 9, 11, 50), _iv(2, 8, 12, 50)])
    assert result.removable_ids == (1,)
    assert result.retained_ids == (2,)
    result = simplify_intervals([_iv(1, 8, 12, 50), _iv(2, 9, 11, 50)])
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)


def test_simplify_zero_lux_intervals_always_removable():
    # 0 lux 区间不改变任何时刻的有效照度（即使无正照度灯），均可撤去
    result = simplify_intervals([_iv(1, 8, 12, 0), _iv(2, 9, 11, 0)])
    assert result.removable_ids == (1, 2)
    assert result.retained_ids == ()
    assert result.original_segments == ()
    assert result.simplified_segments == ()

    # 0 lux 与正照度灯重叠时同样可撤，正照度片段不受影响
    result = simplify_intervals([_iv(1, 8, 12, 0), _iv(2, 8, 12, 50)])
    assert result.removable_ids == (1,)
    assert result.retained_ids == (2,)


def test_simplify_chained_removal_with_updated_set():
    # 以更新后的集合逐条判定：1、2 为同照度副本（08-12），3 独占 12-14（首尾相接）。
    # 降序尝试：3 撤去后 12-14 在原覆盖范围内变暗 → 保留；2 撤去后 1 仍在、照度不变
    # → 撤去；1 在更新集合 {1,3} 上再试，撤去会使 08-12 变暗 → 保留。
    # 并列副本中由 ID 降序决定留最小 ID 的 1。
    result = simplify_intervals([
        _iv(1, 8, 12, 50), _iv(2, 8, 12, 50), _iv(3, 12, 14, 50),
    ])
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1, 3)
    # 精简后两侧恒照度片段同为 08-14 @50（首尾相接同照度合并）
    assert result.original_segments == (Segment(_dt(8), _dt(14), 50),)
    assert result.simplified_segments == (Segment(_dt(8), _dt(14), 50),)


def test_simplify_cross_day_and_cross_year_handled():
    # 跨日：03-01 20:00 → 03-02 10:00 @80 覆盖另一盏同照度灯所在的 08-10
    result = simplify_intervals([
        (1, datetime(2026, 3, 1, 20, tzinfo=timezone.utc),
         datetime(2026, 3, 2, 10, tzinfo=timezone.utc), 80),
        (2, datetime(2026, 3, 2, 8, tzinfo=timezone.utc),
         datetime(2026, 3, 2, 10, tzinfo=timezone.utc), 80),
    ])
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)

    # 跨年：2025-12-31 20:00 → 2026-01-01 10:00 覆盖 00:00-04:00 的同照度灯
    result = simplify_intervals([
        (1, datetime(2025, 12, 31, 20, tzinfo=timezone.utc),
         datetime(2026, 1, 1, 10, tzinfo=timezone.utc), 100),
        (2, datetime(2026, 1, 1, 0, tzinfo=timezone.utc),
         datetime(2026, 1, 1, 4, tzinfo=timezone.utc), 100),
    ])
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)


def test_segments_piecewise_equal_shape_matters():
    # 形状不同（逐时刻不等）：积分相等也判 False；完全相同或同为空判 True
    a = merge_intervals([(_dt(8), _dt(12), 50)])
    b = merge_intervals([(_dt(8), _dt(10), 100)])
    assert segments_piecewise_equal(a, b, _dt(8), _dt(12)) is False
    assert segments_piecewise_equal(a, a, _dt(8), _dt(12)) is True
    assert segments_piecewise_equal([], [], _dt(8), _dt(12)) is True
    # 两侧无灯时刻都按 0：无片段侧与有暗段侧逐时刻相等
    assert segments_piecewise_equal(a, [], _dt(12), _dt(14)) is True
    assert segments_piecewise_equal(a, [], _dt(11), _dt(13)) is False


# ------------------------------------------------------------- 现场照度对账

def test_reconcile_basic_positive_and_negative_deviation():
    # 计划 08-12 @50；实测 08-10 @80（+30）、10-12 @30（-20）；允许偏差 10
    plan = merge_intervals([(_dt(8), _dt(12), 50)])
    readings = [(_dt(8), 80.0), (_dt(10), 30.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 60.0
    assert result.negative_lux_hours == 40.0
    assert result.over_tolerance_hours == 4.0
    assert result.anomalies == (
        DeviationAnomaly(_dt(8), _dt(10), DEVIATION_OVER, 2.0),
        DeviationAnomaly(_dt(10), _dt(12), DEVIATION_UNDER, 2.0),
    )


def test_reconcile_unscheduled_time_counts_as_zero_plan():
    # 窗口内完全无排灯：计划按 0 计算；实测 08-10 @20、10-12 @5，允许偏差 10
    readings = [(_dt(8), 20.0), (_dt(10), 5.0)]
    result = reconcile_illuminance([], readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 50.0
    assert result.negative_lux_hours == 0.0
    assert result.over_tolerance_hours == 2.0  # 仅 08-10 超差
    assert result.anomalies == (DeviationAnomaly(_dt(8), _dt(10), DEVIATION_OVER, 2.0),)


def test_reconcile_exact_tolerance_is_not_anomaly():
    # 偏差恰等于阈值不算异常，但仍计入正负偏差积分
    plan = merge_intervals([(_dt(8), _dt(12), 50)])
    readings = [(_dt(8), 60.0), (_dt(10), 40.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 20.0
    assert result.negative_lux_hours == 20.0
    assert result.over_tolerance_hours == 0.0
    assert result.anomalies == ()


def test_reconcile_merges_adjacent_same_direction_anomalies():
    # 计划 08-14 @50；实测 08-10 @70、10-12 @90、12-14 @45（允许偏差 10）
    # 前两段同为正偏差且相邻 → 合并为 08-12；12-14 实测 45（-5）合规但计入负偏差
    plan = merge_intervals([(_dt(8), _dt(14), 50)])
    readings = [(_dt(8), 70.0), (_dt(10), 90.0), (_dt(12), 45.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(14), 10.0)
    assert result.positive_lux_hours == 20 * 2 + 40 * 2
    assert result.negative_lux_hours == 5 * 2
    assert result.over_tolerance_hours == 4.0
    assert result.anomalies == (DeviationAnomaly(_dt(8), _dt(12), DEVIATION_OVER, 4.0),)


def test_reconcile_plan_change_point_splits_reading_step():
    # 计划 08-10 @50、10-12 @200；实测全程 @100（允许偏差 10）
    # 08-10: +50 超差；10-12: -100 超差 → 两个方向各一段
    plan = merge_intervals([(_dt(8), _dt(10), 50), (_dt(10), _dt(12), 200)])
    readings = [(_dt(8), 100.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 100.0
    assert result.negative_lux_hours == 200.0
    assert result.over_tolerance_hours == 4.0
    assert result.anomalies == (
        DeviationAnomaly(_dt(8), _dt(10), DEVIATION_OVER, 2.0),
        DeviationAnomaly(_dt(10), _dt(12), DEVIATION_UNDER, 2.0),
    )


def test_reconcile_clips_plan_segments_to_window():
    # 跨界计划区间 06-09 @50 裁剪到窗口 08-12；实测 08-12 @40（允许偏差 10）
    plan = merge_intervals([(_dt(6), _dt(9), 50)])
    readings = [(_dt(8), 40.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    # 08-09: -10 恰好到阈值不记异常；09-12: 未排灯计划按 0，实测 +40 超差
    assert result.positive_lux_hours == 120.0
    assert result.negative_lux_hours == 10.0
    assert result.over_tolerance_hours == 3.0
    assert result.anomalies == (DeviationAnomaly(_dt(9), _dt(12), DEVIATION_OVER, 3.0),)
