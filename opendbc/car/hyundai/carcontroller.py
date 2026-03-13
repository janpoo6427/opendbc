import math
import numpy as np
from opendbc.car.carlog import carlog
from opendbc.car.vehicle_model import VehicleModel

try:
  # TODO-SP: We shouldn't really import params from here, but it's the easiest way to get the params for
  #  live tuning temporarily while we understand the angle steering better
  from openpilot.common.params import Params
  PARAMS_AVAILABLE = True
except ImportError:
  carlog.warning("Unable to import Params from openpilot.common.params.")
  PARAMS_AVAILABLE = False

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg, structs
from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance, apply_steer_angle_limits_vm
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.hyundai import hyundaicanfd, hyundaican
from opendbc.car.hyundai.hyundaicanfd import CanBus
from opendbc.car.hyundai.values import HyundaiFlags, Buttons, CarControllerParams, CAR
from opendbc.car.interfaces import CarControllerBase

from opendbc.sunnypilot.car.hyundai.escc import EsccCarController
from opendbc.sunnypilot.car.hyundai.icbm import IntelligentCruiseButtonManagementInterface
from opendbc.sunnypilot.car.hyundai.longitudinal.controller import LongitudinalController
from opendbc.sunnypilot.car.hyundai.lead_data_ext import LeadDataCarController
from opendbc.sunnypilot.car.hyundai.mads import MadsCarController

from opendbc.car.hyundai.torque_reduction_gain import TorqueReductionGainController

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState

# EPS faults if you apply torque while the steering angle is above 90 degrees for more than 1 second
# All slightly below EPS thresholds to avoid fault
MAX_ANGLE = 85
MAX_ANGLE_FRAMES = 89
MAX_ANGLE_CONSECUTIVE_FRAMES = 2

MAX_ANGLE_RATE = 5
ANGLE_SAFETY_BASELINE_MODEL = "KIA_SPORTAGE_HEV_2026"

# ---------------------------------------------------------------------------
# IONIQ 9 전용 스무딩 파라미터 (R-MDPS 특성 반영)
# R-MDPS는 모터가 랙에 직결되어 고주파 진동이 직접 전달되므로
# 저속에서 더 강한 필터링(낮은 alpha)이 필요하다.
# ---------------------------------------------------------------------------
IONIQ9_SMOOTHING_VEGO_MATRIX  = [0.0, 3.0, 8.5, 11.0, 13.8, 22.22, 25.0]
IONIQ9_SMOOTHING_ALPHA_MATRIX = [0.07, 0.09, 0.16, 0.28, 0.52, 0.83, 1.0]
IONIQ9_SMOOTHING_MAX_VEGO     = 25.0   # 90 km/h 이상에서 스무딩 비활성화

# ---------------------------------------------------------------------------
# IONIQ 9 전용 pre-filter 파라미터
# driving model raw output의 고주파 성분을 안전 한계 함수 호출 전에 1차 제거
# ---------------------------------------------------------------------------
IONIQ9_PREFILTER_VEGO_MATRIX  = [0.0,  3.0,  8.5,  13.8, 22.22]
IONIQ9_PREFILTER_ALPHA_MATRIX = [0.12, 0.15, 0.22, 0.38, 0.58]

# ---------------------------------------------------------------------------
# 정차·재출발 구간 토크 스케일 파라미터
# v_ego가 0에 가까울수록 조향 토크를 부드럽게 줄여 정차 직전 핸들 스프링백 방지
# ---------------------------------------------------------------------------
STOP_RAMP_VEGO   = [0.0, 0.3, 0.8, 1.5]   # m/s
STOP_RAMP_SCALE  = [0.0, 0.15, 0.55, 1.0] # torque 비율


def get_baseline_safety_cp():
  from opendbc.car.hyundai.interface import CarInterface
  return CarInterface.get_non_essential_params(ANGLE_SAFETY_BASELINE_MODEL)


def calculate_angle_torque_reduction_gain(params, CS, apply_torque_last, target_torque_reduction_gain):
  """ Calculate the angle torque reduction gain based on the current steering state. """
  target_gain = max(target_torque_reduction_gain, params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN)

  driver_torque = abs(CS.out.steeringTorque)
  alpha = np.interp(driver_torque, [params.STEER_THRESHOLD * .8, params.STEER_THRESHOLD * 2], [0.02, 0.1])

  if CS.out.steeringPressed:
    scale = 100
    clamped_torque_gain = max(apply_torque_last, params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN)
    target_gain = params.ANGLE_MIN_TORQUE_REDUCTION_GAIN + (clamped_torque_gain - params.ANGLE_MIN_TORQUE_REDUCTION_GAIN) \
                  * math.exp(-(driver_torque - params.STEER_THRESHOLD) / scale)
    # IONIQ9 R-MDPS: 운전자 개입 시 알파 상한을 낮춰 토크 스파이크 방지
    alpha = min(alpha, 0.06)
  else:
    # 손 뗀 직후 부드러운 복원을 위해 3단계 보간
    alpha = float(np.interp(driver_torque,
                            [0.0, params.STEER_THRESHOLD * 0.3, params.STEER_THRESHOLD * 0.8],
                            [0.012, 0.025, alpha]))

  # Smooth transition (like a rubber band returning)
  new_gain = apply_torque_last + alpha * (target_gain - apply_torque_last)

  return float(np.clip(new_gain, params.ANGLE_MIN_TORQUE_REDUCTION_GAIN, params.ANGLE_MAX_TORQUE_REDUCTION_GAIN))


def sp_pre_filter_angle(desired_angle: float, pre_filter_last: float, v_ego_raw: float,
                        vego_matrix: list, alpha_matrix: list) -> float:
  """
  1단계 저역통과 필터: driving model raw output의 고주파 jitter를
  안전 한계 함수(apply_steer_angle_limits_vm) 호출 전에 사전 제거한다.

  - 저속(0~3 m/s): alpha ≈ 0.12 → 강한 필터링으로 ±0.3~0.5° jitter 대폭 감쇠
  - 고속(≥22 m/s): alpha ≈ 0.58 → 필터링 약화, 빠른 차선 추종 허용
  - 큰 각도 변화(교차로 복귀, Δ > 3°): alpha를 동적으로 boosting하여 빠른 추종 보장

  Parameters:
    desired_angle     : driving model이 요청한 목표 조향각 (deg)
    pre_filter_last   : 이전 프레임의 pre-filter 출력값 (deg)
    v_ego_raw         : 차량 속도 (m/s)
    vego_matrix       : 속도 보간 포인트 리스트
    alpha_matrix      : alpha 보간 포인트 리스트

  Returns:
    float: 필터링된 목표 조향각 (deg)
  """
  # 속도 기반 기본 alpha 계산
  cutoff_alpha = float(np.interp(v_ego_raw, vego_matrix, alpha_matrix))

  # 교차로 복귀처럼 큰 각도 변화는 필터를 약화시켜 빠른 추종 허용
  # Δ가 클수록 cutoff_alpha를 boosting (최대 0.85까지)
  angle_delta = abs(desired_angle - pre_filter_last)
  if angle_delta > 3.0:
    boost = float(np.interp(angle_delta,
                            [3.0, 6.0, 15.0],
                            [0.0, 0.20, 0.40]))
    cutoff_alpha = float(np.clip(cutoff_alpha + boost, cutoff_alpha, 0.85))

  return float(desired_angle * cutoff_alpha + pre_filter_last * (1.0 - cutoff_alpha))


def sp_smooth_angle(v_ego_raw: float, apply_angle: float, apply_angle_last: float, params) -> float:
  """
  2단계 적응형 스무더: 안전 한계 함수 통과 후 EPS로 전달되기 직전에 적용.

  원본 코드의 문제점:
    1) 안전 한계 함수 이전에 호출되어 효과가 무효화됨 → 안전 한계 후로 이동
    2) |Δangle| ≤ 0.1° skip 로직 → stick-slip 소음 발생 → 제거
    3) 단일 fixed alpha → 교차로 복귀 시 느린 반응 → exponential blend로 개선

  알고리즘:
    base_alpha  : 차량 속도 기반 EMA 알파 (저속 낮음 → 고속 높음)
    alpha_target: 큰 각도 변화 시 수렴할 목표 알파 (0.55)
    k           : 지수 증가 기울기 (1.2, 교차로 복귀 응답성 균형)
    blend       : 1 - exp(-k * angle_delta) → angle_delta가 클수록 1에 수렴
    final_alpha : base_alpha와 alpha_target 사이를 blend로 보간, [0.04, 1.0] 클립

  Parameters:
    v_ego_raw       : 차량 속도 (m/s)
    apply_angle     : 안전 한계 통과 후 목표 조향각 (deg)
    apply_angle_last: 이전 프레임의 최종 조향각 (deg)
    params          : CarControllerParams 인스턴스

  Returns:
    float: 스무딩된 최종 조향각 (deg)
  """
  # 속도 기반 기본 alpha
  base_alpha = float(np.interp(v_ego_raw,
                               params.SMOOTHING_ANGLE_VEGO_MATRIX,
                               params.SMOOTHING_ANGLE_ALPHA_MATRIX))

  angle_delta = abs(apply_angle - apply_angle_last)

  # 지수 함수 기반 연속 blend: Δ가 클수록 alpha_target에 부드럽게 수렴
  # 하드컷(2° 분기) 없이 단일 곡선으로 저속 노이즈 억제 + 교차로 빠른 복귀 동시 달성
  alpha_target = 0.55   # 큰 각도 변화 시 수렴 목표 alpha
  k            = 1.2    # 지수 기울기: 클수록 작은 Δ에서도 빠르게 alpha_target에 도달
  blend        = 1.0 - math.exp(-k * angle_delta)
  final_alpha  = float(np.clip(
    base_alpha + (alpha_target - base_alpha) * blend,
    0.04,   # 하한: EMA가 완전히 멈추지 않도록 보장 (0이면 무한 지연)
    1.0
  ))

  return float(apply_angle * final_alpha + apply_angle_last * (1.0 - final_alpha))


def get_stop_ramp_scale(v_ego_raw: float, lat_active: bool) -> float:
  """
  정차 진입 시 조향 토크를 부드럽게 줄여 핸들 스프링백 방지.

  v_ego가 0에 가까워질수록 torque scale을 0으로 ramping down하여
  openpilot이 정차 직전에 토크를 갑자기 끊을 때 발생하는
  핸들 좌우 튕김(spring back) 현상을 억제한다.

  Parameters:
    v_ego_raw  : 차량 속도 (m/s)
    lat_active : 횡방향 제어 활성 여부

  Returns:
    float: 0.0 ~ 1.0 사이의 토크 스케일 팩터
  """
  if not lat_active:
    return 0.0
  return float(np.interp(v_ego_raw, STOP_RAMP_VEGO, STOP_RAMP_SCALE))


def process_hud_alert(enabled, fingerprint, hud_control):
  sys_warning = (hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw))

  # initialize to no line visible
  # TODO: this is not accurate for all cars
  sys_state = 1
  if hud_control.leftLaneVisible and hud_control.rightLaneVisible or sys_warning:  # HUD alert only display when LKAS status is active
    sys_state = 3 if enabled or sys_warning else 4
  elif hud_control.leftLaneVisible:
    sys_state = 5
  elif hud_control.rightLaneVisible:
    sys_state = 6

  # initialize to no warnings
  left_lane_warning = 0
  right_lane_warning = 0
  if hud_control.leftLaneDepart:
    left_lane_warning = 1 if fingerprint in (CAR.GENESIS_G90, CAR.GENESIS_G80) else 2
  if hud_control.rightLaneDepart:
    right_lane_warning = 1 if fingerprint in (CAR.GENESIS_G90, CAR.GENESIS_G80) else 2

  return sys_warning, sys_state, left_lane_warning, right_lane_warning


def parse_tq_rdc_gain(val):
  """
  Returns the float value divided by 100 if val is not None, else returns None.
  """
  if val is not None:
    return float(val) / 100
  return None


def parse_scaled_value(val, scale=10):
  if val is not None:
    return float(val) / scale
  return None


class CarController(CarControllerBase, EsccCarController, LeadDataCarController, LongitudinalController, MadsCarController,
                    IntelligentCruiseButtonManagementInterface):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    EsccCarController.__init__(self, CP, CP_SP)
    MadsCarController.__init__(self)
    LeadDataCarController.__init__(self, CP)
    LongitudinalController.__init__(self, CP, CP_SP)
    IntelligentCruiseButtonManagementInterface.__init__(self, CP, CP_SP)
    self.CAN = CanBus(CP)
    self.params = CarControllerParams(CP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.angle_limit_counter = 0

    # Vehicle model used for lateral limiting
    self.VM = VehicleModel(CP)
    self.BASELINE_VM = VehicleModel(get_baseline_safety_cp())

    self.accel_last = 0
    self.apply_torque_last = 0
    self.car_fingerprint = CP.carFingerprint
    self.last_button_frame = 0

    self.apply_angle_last = 0
    self.angle_torque_reduction_gain = 0

    # pre-filter 상태 변수: driving model raw output을 안전 한계 전에 1차 필터링
    self.pre_filter_angle_last = 0.0

    # For future parametrization / tuning
    self.angle_enable_smoothing_factor = True

    # ---------------------------------------------------------------------------
    # IONIQ 9 감지 및 파라미터 덮어쓰기 (초기화 시 1회만 수행)
    # is_ioniq9 런타임 플래그 없이 self.params를 직접 교체하므로
    # update()와 sp_smooth_angle()에서 분기 없이 동일 코드가 동작한다.
    # ---------------------------------------------------------------------------
    if "IONIQ_9" in str(CP.carFingerprint).upper():
      self.params.SMOOTHING_ANGLE_VEGO_MATRIX  = IONIQ9_SMOOTHING_VEGO_MATRIX
      self.params.SMOOTHING_ANGLE_ALPHA_MATRIX = IONIQ9_SMOOTHING_ALPHA_MATRIX
      self.params.SMOOTHING_ANGLE_MAX_VEGO     = IONIQ9_SMOOTHING_MAX_VEGO
      self._prefilter_vego  = IONIQ9_PREFILTER_VEGO_MATRIX
      self._prefilter_alpha = IONIQ9_PREFILTER_ALPHA_MATRIX
      carlog.info("CarController: IONIQ9 R-MDPS 전용 스무딩 파라미터 적용")
    else:
      # 일반 차종은 기존 파라미터를 그대로 사용하되,
      # pre-filter도 기본 스무딩 매트릭스 기반으로 동작
      self._prefilter_vego  = list(CarControllerParams.SMOOTHING_ANGLE_VEGO_MATRIX)
      self._prefilter_alpha = [a * 0.6 for a in CarControllerParams.SMOOTHING_ANGLE_ALPHA_MATRIX]

    self._params = Params() if PARAMS_AVAILABLE else None
    if PARAMS_AVAILABLE:
      self.params.ANGLE_MIN_TORQUE_REDUCTION_GAIN = parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleMinTorqueReductionGain")) or self.params.ANGLE_MIN_TORQUE_REDUCTION_GAIN

      self.params.ANGLE_MAX_TORQUE_REDUCTION_GAIN = parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleMaxTorqueReductionGain")) or self.params.ANGLE_MAX_TORQUE_REDUCTION_GAIN

      self.params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN = parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleActiveTorqueReductionGain")) or self.params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN

      self.params.ANGLE_TORQUE_OVERRIDE_CYCLES = int(self._params.get("HkgTuningOverridingCycles") or self.params.ANGLE_TORQUE_OVERRIDE_CYCLES)
      self.angle_enable_smoothing_factor = self._params.get_bool("EnableHkgTuningAngleSmoothingFactor")

    self.angle_torque_reduction_gain_controller = TorqueReductionGainController(
      angle_threshold=.3,
      debounce_time=.1,
      min_gain=self.params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN,
      max_gain=self.params.ANGLE_MAX_TORQUE_REDUCTION_GAIN,
      ramp_up_rate=self.params.ANGLE_RAMP_UP_TORQUE_REDUCTION_RATE,
      ramp_down_rate=self.params.ANGLE_RAMP_DOWN_TORQUE_REDUCTION_RATE
    )

  def update(self, CC, CC_SP, CS, now_nanos):
    EsccCarController.update(self, CS)
    LeadDataCarController.update(self, CC_SP)
    MadsCarController.update(self, self.CP, CC, CC_SP, self.frame)
    if self.frame % 5 == 0:
      LongitudinalController.update(self, CC, CS)

    actuators = CC.actuators
    hud_control = CC.hudControl

    v_ego_raw = CS.out.vEgoRaw

    # ---------------------------------------------------------------------------
    # common_fault_avoidance: 토크/앵글 조향 모두에 적용되는 공통 안전 가드
    # 조향각이 MAX_ANGLE(85°) 이상인 상태가 MAX_ANGLE_FRAMES(89프레임) 이상
    # 지속되면 lat_active를 False로 전환하여 EPS 폴트를 예방한다.
    # 토크 조향과 앵글 조향 양쪽 모두 이 필터링된 값을 사용한다.
    # ---------------------------------------------------------------------------
    self.angle_limit_counter, lat_active_filtered = common_fault_avoidance(
      abs(CS.out.steeringAngleDeg) >= MAX_ANGLE,
      CC.latActive,
      self.angle_limit_counter,
      MAX_ANGLE_FRAMES,
      MAX_ANGLE_CONSECUTIVE_FRAMES
    )

    # steering torque
    if not self.CP.flags & HyundaiFlags.CANFD_ANGLE_STEERING:
      new_torque = int(round(actuators.torque * self.params.STEER_MAX))
      apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.params)
      apply_steer_req = lat_active_filtered

    # angle control
    else:
      desired_angle = np.clip(actuators.steeringAngleDeg, -self.params.ANGLE_LIMITS.STEER_ANGLE_MAX, self.params.ANGLE_LIMITS.STEER_ANGLE_MAX)

      # -----------------------------------------------------------------------
      # [1단계] Pre-filter: driving model raw output의 고주파 jitter를 사전 제거
      # 안전 한계 함수(apply_steer_angle_limits_vm) 호출 전에 적용하여
      # 노이즈가 포함된 raw 각도가 한계 함수에 들어가는 것을 방지.
      # lat_active_filtered일 때만 업데이트하여 비활성 시 현재 실제 각도를 추적.
      # -----------------------------------------------------------------------
      if lat_active_filtered and self.angle_enable_smoothing_factor:
        desired_angle = sp_pre_filter_angle(
          desired_angle,
          self.pre_filter_angle_last,
          v_ego_raw,
          self._prefilter_vego,
          self._prefilter_alpha
        )
        self.pre_filter_angle_last = desired_angle
      else:
        # 비활성 구간: pre_filter_last를 실제 조향각에 부드럽게 추적
        # (재활성화 시 급격한 초기값 점프 방지)
        self.pre_filter_angle_last += 0.10 * (CS.out.steeringAngleDeg - self.pre_filter_angle_last)

      # -----------------------------------------------------------------------
      # [2단계] 안전 한계 적용 (VM 기반)
      # pre-filter 통과 후의 desired_angle에 적용하므로 노이즈가 최소화된 상태
      # -----------------------------------------------------------------------
      apply_angle = apply_steer_angle_limits_vm(
        desired_angle, self.apply_angle_last, v_ego_raw,
        CS.out.steeringAngleDeg, lat_active_filtered, self.params, self.VM
      )

      # -----------------------------------------------------------------------
      # if we are not the baseline model, we use the baseline model for further
      # limits to prevent a panda block since it is hardcoded for baseline model.
      # -----------------------------------------------------------------------
      if self.CP.carFingerprint != ANGLE_SAFETY_BASELINE_MODEL:
        apply_angle = apply_steer_angle_limits_vm(
          apply_angle if apply_angle is not None else desired_angle,
          self.apply_angle_last, v_ego_raw,
          CS.out.steeringAngleDeg, lat_active_filtered,
          self.params, self.BASELINE_VM
        )

      # -----------------------------------------------------------------------
      # [3단계] 토크 감소 게인 계산
      # 조향각 포화도 기반 TorqueReductionGainController로 목표 게인 산출 후
      # calculate_angle_torque_reduction_gain으로 부드러운 전환 처리.
      # Failsafe(apply_angle is None) 전에 계산하여 torque를 0으로 설정.
      # -----------------------------------------------------------------------
      target_torque_reduction_gain = self.angle_torque_reduction_gain_controller.update(
        last_requested_angle=self.apply_angle_last,
        actual_angle=CS.out.steeringAngleDeg,
        lat_active=lat_active_filtered
      )

      # This method ensures that the torque gives up when overriding and controls the ramp rate to avoid feeling jittery.
      apply_torque = calculate_angle_torque_reduction_gain(self.params, CS, self.apply_torque_last, target_torque_reduction_gain)

      # -----------------------------------------------------------------------
      # Failsafe: 안전 한계 위반 감지 시 토크/각도 즉시 초기화
      # -----------------------------------------------------------------------
      if apply_angle is None:
        apply_torque = 0
        apply_angle = CS.out.steeringAngleDeg
        apply_steer_req = False
      else:
        # -----------------------------------------------------------------------
        # [4단계] 정차 진입 토크 램프: 핸들 스프링백 방지
        # v_ego가 0에 가까워질수록 토크를 부드럽게 줄여
        # openpilot이 정차 직전 토크를 끊을 때 핸들이 튕기는 현상 억제.
        # -----------------------------------------------------------------------
        #stop_scale = get_stop_ramp_scale(v_ego_raw, lat_active_filtered)
        #apply_torque = float(apply_torque) * stop_scale

        # -----------------------------------------------------------------------
        # [5단계] 2단계 스무더: 안전 한계 통과 후 EPS 전달 직전에 최종 스무딩
        # sp_smooth_angle은 반드시 apply_steer_angle_limits_vm 이후에 호출해야
        # 안전 한계가 스무딩을 덮어쓰는 문제가 발생하지 않는다.
        # -----------------------------------------------------------------------
        if self.angle_enable_smoothing_factor and abs(v_ego_raw) < self.params.SMOOTHING_ANGLE_MAX_VEGO:
          apply_angle = sp_smooth_angle(v_ego_raw, apply_angle, self.apply_angle_last, self.params)

        # apply_steer_req is True when we are actively attempting to steer and under the angle limit. Otherwise the user is overriding.
        apply_steer_req = lat_active_filtered and apply_torque != 0

      # After we've used the last angle wherever we needed it, we now update it.
      # 스무딩 적용 후의 값을 저장하므로 다음 프레임의 apply_angle_last는
      # 항상 EPS에 실제로 전달된 부드러운 값이 된다.
      self.apply_angle_last = apply_angle

    if not CC.latActive:
      apply_torque = 0

    # Hold torque with induced temporary fault when cutting the actuation bit
    # FIXME: we don't use this with CAN FD?
    torque_fault = CC.latActive and not apply_steer_req

    self.apply_torque_last = apply_torque

    # accel + longitudinal
    accel = float(np.clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
    stopping = actuators.longControlState == LongCtrlState.stopping
    set_speed_in_units = hud_control.setSpeed * (CV.MS_TO_KPH if CS.is_metric else CV.MS_TO_MPH)

    can_sends = []

    # *** common hyundai stuff ***

    # tester present - w/ no response (keeps relevant ECU disabled)
    if self.frame % 100 == 0 and not ((self.CP.flags & HyundaiFlags.CANFD_CAMERA_SCC) or self.ESCC.enabled) and \
            self.CP.openpilotLongitudinalControl:
      # for longitudinal control, either radar or ADAS driving ECU
      addr, bus = 0x7d0, self.CAN.ECAN if self.CP.flags & HyundaiFlags.CANFD else 0
      if self.CP.flags & HyundaiFlags.CANFD_LKA_STEERING.value:
        addr, bus = 0x730, self.CAN.ECAN
      can_sends.append(make_tester_present_msg(addr, bus, suppress_response=True))

      # for blinkers
      if self.CP.flags & HyundaiFlags.ENABLE_BLINKERS:
        can_sends.append(make_tester_present_msg(0x7b1, self.CAN.ECAN, suppress_response=True))

    # *** CAN/CAN FD specific ***
    if self.CP.flags & HyundaiFlags.CANFD:
      can_sends.extend(self.create_canfd_msgs(apply_steer_req, apply_torque, set_speed_in_units, accel,
                                              stopping, hud_control, CS, CC))
    else:
      can_sends.extend(self.create_can_msgs(apply_steer_req, apply_torque, torque_fault, set_speed_in_units, accel,
                                            stopping, hud_control, actuators, CS, CC))

    # Intelligent Cruise Button Management
    can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CS, CC_SP, self.packer, self.frame, self.last_button_frame, self.CAN))

    new_actuators = actuators.as_builder()
    new_actuators.torque = apply_torque / self.params.STEER_MAX
    new_actuators.torqueOutputCan = apply_torque
    new_actuators.steeringAngleDeg = self.apply_angle_last
    new_actuators.accel = self.tuning.actual_accel

    self.frame += 1
    return new_actuators, can_sends

  def create_can_msgs(self, apply_steer_req, apply_torque, torque_fault, set_speed_in_units, accel, stopping, hud_control, actuators, CS, CC):
    can_sends = []

    # HUD messages
    sys_warning, sys_state, left_lane_warning, right_lane_warning = process_hud_alert(CC.enabled, self.car_fingerprint,
                                                                                      hud_control)

    can_sends.append(hyundaican.create_lkas11(self.packer, self.frame, self.CP, apply_torque, apply_steer_req,
                                              torque_fault, CS.lkas11, sys_warning, sys_state, CC.enabled,
                                              hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                              left_lane_warning, right_lane_warning,
                                              self.lkas_icon))

    # Button messages
    if not self.CP.openpilotLongitudinalControl:
      if CC.cruiseControl.cancel:
        can_sends.append(hyundaican.create_clu11(self.packer, self.frame, CS.clu11, Buttons.CANCEL, self.CP))
      elif CC.cruiseControl.resume:
        # send resume at a max freq of 10Hz
        if (self.frame - self.last_button_frame) * DT_CTRL > 0.1:
          # send 25 messages at a time to increases the likelihood of resume being accepted
          can_sends.extend([hyundaican.create_clu11(self.packer, self.frame, CS.clu11, Buttons.RES_ACCEL, self.CP)] * 25)
          if (self.frame - self.last_button_frame) * DT_CTRL >= 0.15:
            self.last_button_frame = self.frame

    if self.frame % 2 == 0 and self.CP.openpilotLongitudinalControl:
      # TODO: unclear if this is needed
      jerk = 3.0 if actuators.longControlState == LongCtrlState.pid else 1.0
      use_fca = self.CP.flags & HyundaiFlags.USE_FCA.value
      can_sends.extend(hyundaican.create_acc_commands(self.packer, CC.enabled, accel, jerk, int(self.frame / 2),
                                                      self.lead_data, hud_control, set_speed_in_units, stopping,
                                                      CC.cruiseControl.override, use_fca, self.CP,
                                                      CS.main_cruise_enabled, self.tuning, self.ESCC))

    # 20 Hz LFA MFA message
    if self.frame % 5 == 0 and self.CP.flags & HyundaiFlags.SEND_LFA.value:
      can_sends.append(hyundaican.create_lfahda_mfc(self.packer, CC.enabled, self.lfa_icon))

    # 5 Hz ACC options
    if self.frame % 20 == 0 and self.CP.openpilotLongitudinalControl:
      can_sends.extend(hyundaican.create_acc_opt(self.packer, self.CP, self.ESCC))

    # 2 Hz front radar options
    if self.frame % 50 == 0 and self.CP.openpilotLongitudinalControl and not self.ESCC.enabled:
      can_sends.append(hyundaican.create_frt_radar_opt(self.packer))

    return can_sends

  def create_canfd_msgs(self, apply_steer_req, apply_torque, set_speed_in_units, accel, stopping, hud_control, CS, CC):
    can_sends = []

    lka_steering = self.CP.flags & HyundaiFlags.CANFD_LKA_STEERING
    lka_steering_long = lka_steering and self.CP.openpilotLongitudinalControl

    # steering control
    can_sends.extend(hyundaicanfd.create_steering_messages(self.packer, self.CP, self.CAN, CC.enabled, apply_steer_req, apply_torque, self.apply_angle_last
                                                           , self.lkas_icon))

    # prevent LFA from activating on LKA steering cars by sending "no lane lines detected" to ADAS ECU
    if self.frame % 5 == 0 and lka_steering:
      can_sends.append(hyundaicanfd.create_suppress_lfa(self.packer, self.CAN, CS.lfa_block_msg,
                                                        self.CP.flags & HyundaiFlags.CANFD_LKA_STEERING_ALT))

    # LFA and HDA icons
    if self.frame % 5 == 0 and (not lka_steering or lka_steering_long):
      can_sends.append(hyundaicanfd.create_lfahda_cluster(self.packer, self.CAN, CC.enabled, self.lfa_icon))

    # blinkers
    if lka_steering and self.CP.flags & HyundaiFlags.ENABLE_BLINKERS:
      can_sends.extend(hyundaicanfd.create_spas_messages(self.packer, self.CAN, CC.leftBlinker, CC.rightBlinker))

    if self.CP.openpilotLongitudinalControl:
      if lka_steering:
        can_sends.extend(hyundaicanfd.create_adrv_messages(self.packer, self.CAN, self.frame))
      else:
        can_sends.extend(hyundaicanfd.create_fca_warning_light(self.packer, self.CAN, self.frame))
      if self.frame % 2 == 0:
        can_sends.append(hyundaicanfd.create_acc_control(self.packer, self.CAN, CC.enabled, self.accel_last, accel, stopping, CC.cruiseControl.override,
                                                         set_speed_in_units, hud_control, self.lead_data, CS.main_cruise_enabled, self.tuning))
        self.accel_last = accel
    else:
      # button presses
      if (self.frame - self.last_button_frame) * DT_CTRL > 0.25:
        # cruise cancel
        if CC.cruiseControl.cancel:
          if self.CP.flags & HyundaiFlags.CANFD_ALT_BUTTONS:
            can_sends.append(hyundaicanfd.create_acc_cancel(self.packer, self.CP, self.CAN, CS.cruise_info))
            self.last_button_frame = self.frame
          else:
            for _ in range(20):
              can_sends.append(hyundaicanfd.create_buttons(self.packer, self.CP, self.CAN, CS.buttons_counter + 1, Buttons.CANCEL))
            self.last_button_frame = self.frame

        # cruise standstill resume
        elif CC.cruiseControl.resume:
          if self.CP.flags & HyundaiFlags.CANFD_ALT_BUTTONS:
            # TODO: resume for alt button cars
            pass
          else:
            for _ in range(20):
              can_sends.append(hyundaicanfd.create_buttons(self.packer, self.CP, self.CAN, CS.buttons_counter + 1, Buttons.RES_ACCEL))
            self.last_button_frame = self.frame

    return can_sends