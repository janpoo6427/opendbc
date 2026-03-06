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


def get_baseline_safety_cp():
  from opendbc.car.hyundai.interface import CarInterface
  return CarInterface.get_non_essential_params(ANGLE_SAFETY_BASELINE_MODEL)

def calculate_angle_torque_reduction_gain(params, CS, apply_gain_last, target_torque_reduction_gain):
  """ 
  Natural torque control based on pure driver intent and physical limits.
  
  Philosophy: Remove all artificial angle-based damping
  Focus only on: 1) Driver safety 2) Smooth transitions 3) Physical limits
  
  Args:
    apply_gain_last: Previous gain value (0.0 ~ 1.0)
    target_torque_reduction_gain: Target gain from controller
  
  Returns:
    float: New gain value (0.0 ~ 1.0)
  """
  
  # 초기값 보장 - 비정상적인 값 방지
  if apply_gain_last is None or apply_gain_last <= 0:
    apply_gain_last = params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN
  
  # [1] 기본 게인 설정
  target_gain = max(target_torque_reduction_gain, params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN)

  driver_torque = abs(CS.out.steeringTorque)
  alpha = np.interp(driver_torque, [params.STEER_THRESHOLD * .8, params.STEER_THRESHOLD * 2], [0.05, 0.2])

  # [2] 운전자 개입 시 토크 감쇠 (안전 필수)
  if CS.out.steeringPressed:
    scale = 100
    clamped_torque_gain = max(apply_gain_last, params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN)
    target_gain = params.ANGLE_MIN_TORQUE_REDUCTION_GAIN + (clamped_torque_gain - params.ANGLE_MIN_TORQUE_REDUCTION_GAIN) \
                  * math.exp(-(driver_torque - params.STEER_THRESHOLD) / scale)
  
  # [3] 인위적인 각도 기반 감쇠 완전 제거
  # 조향 방향과 무관하게 EPS가 최대 성능을 발휘하도록 함
  
  # [4] 속도별 자연스러운 토크 변화 관리
  v_ego = abs(CS.out.vEgoRaw)
  
  if not CS.out.steeringPressed:
    torque_change = abs(target_gain - apply_gain_last)
    
    # 속도에 따른 허용 변화량 (물리적 한계 고려)
    max_changes = [x * params.ANGLE_TORQUE_GAIN_RECOVERY_SCALE for x in [0.02, 0.15, 0.25]]

    max_change = np.interp(v_ego, [1.0, 10.0, 30.0], [0.02, 0.15, 0.25])
    
    if torque_change > max_change * 2:
      if target_gain > apply_gain_last:
        target_gain = min(target_gain, apply_gain_last + max_change)
      else:
        target_gain = max(target_gain, apply_gain_last - max_change)
        
    # 저속에서 부드러운 전환 (정차 떨림 방지)
    if v_ego < 2.0:
      alpha *= 0.3

  # [5] 최종 적용
  new_gain = apply_gain_last + alpha * (target_gain - apply_gain_last)

  return float(np.clip(new_gain, params.ANGLE_MIN_TORQUE_REDUCTION_GAIN, params.ANGLE_MAX_TORQUE_REDUCTION_GAIN))



def sp_smooth_angle(v_ego_raw: float, apply_angle: float, apply_angle_last: float) -> float:
  """
  Unified continuous steering control with pure error-based response.
  
  Design Philosophy: "Steering is just error correction"
  - Large errors → Fast response (turning AND returning equally)
  - Small errors → Hunting analysis → Selective control
  - No artificial distinction between turning modes
  
  Core Innovation: Pure magnitude-based reactivity without directional bias
  """
  
  # dt 값을 함수 시작 시 한 번만 계산 (중복 제거)
  try:
    dt = DT_CTRL if DT_CTRL > 0 else 0.01
  except NameError:
    dt = 0.01
  
  angle_diff = abs(apply_angle - apply_angle_last)
  
  # [1] 물리적 안정성 (정차 시 제어 중단)
  if abs(v_ego_raw) < 0.5 and angle_diff < 0.2:
    return apply_angle_last
  
  # [2] 노이즈 필터링 (MDPS 소음 방지)
  if angle_diff < CarControllerParams.ANTI_HUNTING_THRESHOLD: # 0.03:
    return apply_angle_last
  
  # [3] 순수한 에러 크기 기반 반응성 결정 (핵심 철학)
  if angle_diff > 0.5:
    # 대형 조향 (급커브 진입/탈출, 큰 복귀 등)
    reactivity_factor = 4.0 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE
  elif angle_diff > 0.1:
    # 일반 조향 (차선 유지, 완만한 조정)
    reactivity_factor = np.interp(angle_diff, [0.1, 0.5 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE], [2.5, 4.0 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE])
  else:
    # 미세 조향 (헌팅 가능성 구간)
    angle_change_rate = angle_diff / dt
    
    # 작은 변화에서도 속도가 빠르면 정상 조향으로 판단
    magnitude_factor = np.interp(angle_diff, [0.0, 0.05, 0.1], [0.6 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE, 1.0 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE, 2.5 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE])
    rate_factor = np.interp(angle_change_rate, [0.0, 3.0, 10.0], [0.7 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE, 1.3 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE, 2.2 * CarControllerParams.ANGLE_STEERING_REACTIVITY_SCALE])
    reactivity_factor = magnitude_factor * rate_factor

  # [4] 반응성 기반 적응형 스무딩
  if angle_diff > 0.08:
    try:
      smoothing_vego = CarControllerParams.SMOOTHING_ANGLE_VEGO_MATRIX
      smoothing_alpha = CarControllerParams.SMOOTHING_ANGLE_ALPHA_MATRIX
    except AttributeError:
      smoothing_vego = [0, 8.5, 11, 13.8, 22.22]
      smoothing_alpha = [0.05, 0.1, 0.3, 0.6, 1]
    
    base_alpha = np.interp(v_ego_raw, smoothing_vego, smoothing_alpha)
    adjusted_alpha = float(np.clip(base_alpha * reactivity_factor, 0.2, 1.0))
    
    smoothed_angle = (apply_angle * adjusted_alpha) + (apply_angle_last * (1.0 - adjusted_alpha))
  else:
    smoothed_angle = apply_angle
  
  # [5] 통합 Rate Limiter (방향 무관)


  # 속도별 조향 속도 제한값에 일괄 적용
  base_rates = [x * CarControllerParams.ANGLE_MAX_STEERING_RATE_SCALE for x in [8.0, 12.0, 16.0, 20.0]]
  base_rate = np.interp(abs(v_ego_raw), [0., 10., 20., 30.], base_rates)
  effective_max_rate = base_rate * np.clip(reactivity_factor, 1.0, 3.5)
  
  max_delta = effective_max_rate * dt
  
  delta = smoothed_angle - apply_angle_last
  final_angle = apply_angle_last + np.clip(delta, -max_delta, max_delta)
  
  return final_angle


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
    
    # 변수 분리: 토크 모드와 각도 모드의 서로 다른 데이터 타입 처리
    self.apply_torque_last = 0      # Legacy 토크 모드: 실제 토크값 (0~4096)
    self.apply_gain_last = 1.0      # CAN FD 각도 모드: 게인값 (0.0~1.0)
    
    self.car_fingerprint = CP.carFingerprint
    self.last_button_frame = 0

    self.apply_angle_last = 0
    self.angle_torque_reduction_gain = 0

    # For future parametrization / tuning
    self.angle_enable_smoothing_factor = True

    self.frame = 0

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


      self.params.ANGLE_ANTI_HUNTING_THRESHOLD = parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleAntiHuntingThreshold")) or self.params.ANGLE_ANTI_HUNTING_THRESHOLD
      
      self.params.ANGLE_STEERING_REACTIVITY_SCALE = parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleSteeringReactivityScale")) or self.params.ANGLE_STEERING_REACTIVITY_SCALE

      self.params.ANGLE_MAX_STEERING_RATE_SCALE = parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleMaxSteeringRateScale")) or self.params.ANGLE_MAX_STEERING_RATE_SCALE
      self.params.ANGLE_TORQUE_GAIN_RECOVERY_SCALE = parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleTorqueGainRecoveryScale")) or self.params.ANGLE_TORQUE_GAIN_RECOVERY_SCALE
      
      # 파라미터 변경 시 게인 관련 내부 상태 동기화
      self.apply_gain_last = self.params.ANGLE_ACTIVE_TORQUE_REDUCTION_GAIN

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

    # 변수 초기화 (안전장치)
    apply_steer_req = False
    apply_torque = 0
    torque_fault = False

    # steering torque
    if not self.CP.flags & HyundaiFlags.CANFD_ANGLE_STEERING:
      # ===== Legacy 토크 기반 제어 =====
      self.angle_limit_counter, apply_steer_req = common_fault_avoidance(
          abs(CS.out.steeringAngleDeg) >= MAX_ANGLE, CC.latActive,
          self.angle_limit_counter, MAX_ANGLE_FRAMES,
          MAX_ANGLE_CONSECUTIVE_FRAMES)
      new_torque = int(round(actuators.torque * self.params.STEER_MAX))
      apply_torque = apply_driver_steer_torque_limits(
          new_torque, self.apply_torque_last, CS.out.steeringTorque, self.params)
      
      # 토크 기반 제어에서는 실제 토크값 저장
      self.apply_torque_last = apply_torque
      
      # Hold torque with induced temporary fault when cutting the actuation bit
      torque_fault = CC.latActive and not apply_steer_req

    else:
      # ===== CAN FD 각도 기반 제어 =====
      v_ego_raw = CS.out.vEgoRaw
      desired_angle = np.clip(
          actuators.steeringAngleDeg, 
          -self.params.ANGLE_LIMITS.STEER_ANGLE_MAX, 
          self.params.ANGLE_LIMITS.STEER_ANGLE_MAX)

      if self.angle_enable_smoothing_factor and abs(v_ego_raw) < CarControllerParams.SMOOTHING_ANGLE_MAX_VEGO:
        desired_angle = sp_smooth_angle(v_ego_raw, desired_angle, self.apply_angle_last)

      apply_angle = apply_steer_angle_limits_vm(
          desired_angle, self.apply_angle_last, v_ego_raw, 
          CS.out.steeringAngleDeg, CC.latActive, self.params, self.VM)

      if self.CP.carFingerprint != ANGLE_SAFETY_BASELINE_MODEL:
        apply_angle = apply_steer_angle_limits_vm(
            apply_angle or desired_angle, self.apply_angle_last, v_ego_raw, 
            CS.out.steeringAngleDeg, CC.latActive, self.params, self.BASELINE_VM)

      target_torque_reduction_gain = self.angle_torque_reduction_gain_controller.update(
        last_requested_angle=self.apply_angle_last,
        actual_angle=CS.out.steeringAngleDeg,
        lat_active=CC.latActive
      )

      # ★ 핵심 수정: 올바른 변수 전달
      apply_torque = calculate_angle_torque_reduction_gain(
          self.params, CS, 
          self.apply_gain_last,  # ← 게인값 전달 (0.0~1.0)
          target_torque_reduction_gain)

      # 각도 기반 제어에서는 게인값 저장
      self.apply_gain_last = apply_torque

      apply_steer_req = CC.latActive and apply_torque > 0

      if apply_angle is None:
        apply_torque = 0
        apply_angle = CS.out.steeringAngleDeg
        apply_steer_req = False

      self.apply_angle_last = apply_angle
      torque_fault = False

    if not CC.latActive:
      apply_torque = 0

    # accel + longitudinal
    accel = float(np.clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
    stopping = actuators.longControlState == LongCtrlState.stopping
    set_speed_in_units = hud_control.setSpeed * (CV.MS_TO_KPH if CS.is_metric else CV.MS_TO_MPH)

    can_sends = []

    # *** common hyundai stuff ***
    if self.frame % 100 == 0 and not ((self.CP.flags & HyundaiFlags.CANFD_CAMERA_SCC) or self.ESCC.enabled) and \
            self.CP.openpilotLongitudinalControl:
      addr, bus = 0x7d0, self.CAN.ECAN if self.CP.flags & HyundaiFlags.CANFD else 0
      if self.CP.flags & HyundaiFlags.CANFD_LKA_STEERING.value:
        addr, bus = 0x730, self.CAN.ECAN
      can_sends.append(make_tester_present_msg(addr, bus, suppress_response=True))

      if self.CP.flags & HyundaiFlags.ENABLE_BLINKERS:
        can_sends.append(make_tester_present_msg(0x7b1, self.CAN.ECAN, suppress_response=True))

    # *** CAN/CAN FD specific ***
    if self.CP.flags & HyundaiFlags.CANFD:
      can_sends.extend(self.create_canfd_msgs(apply_steer_req, apply_torque, set_speed_in_units, accel,
                                              stopping, hud_control, CS, CC))
    else:
      can_sends.extend(self.create_can_msgs(apply_steer_req, apply_torque, torque_fault, set_speed_in_units, accel,
                                            stopping, hud_control, actuators, CS, CC))

    can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CS, CC_SP, self.packer, self.frame, self.last_button_frame, self.CAN))

    new_actuators = actuators.as_builder()
    
    # ★ 모드별 분리 처리
    if self.CP.flags & HyundaiFlags.CANFD_ANGLE_STEERING:
      # 각도 모드: apply_torque는 게인값이므로 그대로 사용
      new_actuators.torque = apply_torque
      new_actuators.torqueOutputCan = apply_torque
      new_actuators.steeringAngleDeg = self.apply_angle_last
    else:
      # 토크 모드: 정규화된 토크값 사용
      new_actuators.torque = apply_torque / self.params.STEER_MAX
      new_actuators.torqueOutputCan = apply_torque
      new_actuators.steeringAngleDeg = CS.out.steeringAngleDeg
    
    new_actuators.accel = accel

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
