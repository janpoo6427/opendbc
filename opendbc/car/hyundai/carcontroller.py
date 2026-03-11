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

  # Smooth transition (like a rubber band returning)
  new_gain = apply_torque_last + alpha * (target_gain - apply_torque_last)

  return float(np.clip(new_gain, params.ANGLE_MIN_TORQUE_REDUCTION_GAIN, params.ANGLE_MAX_TORQUE_REDUCTION_GAIN))


def sp_smooth_angle(v_ego_raw: float, apply_angle: float, apply_angle_last: float) -> float:
  """
  Smooth the steering angle change based on vehicle speed and an optional smoothing offset.

  This function helps prevent abrupt steering changes by blending the new desired angle (`apply_angle`)
  with the previously applied angle (`apply_angle_last`). The blend factor (alpha) is dynamically calculated
  based on the vehicle's current speed using a predefined lookup table.

  Behavior:
    - At low speeds, the smoothing is strong, keeping the steering more stable.
    - At higher speeds, the smoothing is relaxed, allowing quicker responses.
    - If the angle change is negligible (≤ 0.1 deg), smoothing is skipped for responsiveness.

  Parameters:
    v_ego_raw (float): Raw vehicle speed in m/s.
    apply_angle (float): New target steering angle in degrees.
    apply_angle_last (float): Previously applied steering angle in degrees.

  Returns:
    float: Smoothed steering angle.
  """
  #1. 이번 프레임의 조향 오차(Error)
  error = abs(apply_angle - apply_angle_last)
  #2. 속도 기반 기본 하한선 (너무 잠기지 않게 최소 0.08 보장)
  base_alpha = float(np.interp(v_ego_raw, 
                               CarControllerParams.SMOOTHING_ANGLE_VEGO_MATRIX, 
                               CarControllerParams.SMOOTHING_ANGLE_ALPHA_MATRIX))
  base_alpha = max(base_alpha, 0.4)
  #3. [핵심 1] 속도 비례 탄젠트 민감도 (사용자 아이디어)
  #- 저속(0m/s): 1.0 (곡선이 완만해져 미세 오차를 무시함)
  #- 고속(15m/s~): 5.0 (곡선이 가파라져 미세 오차에도 즉각 반응)
  sensitivity = float(np.interp(v_ego_raw, [0.0, 5.0, 15.0], [0.3, 2.5, 5.0]))
  #4. [핵심 2] 속도 비례 안전 천장 (저속 소음 최후의 방어막)
  #20Hz 노이즈 때문에 저속에서는 아무리 크게 꺾여도 35% 이상 필터를 열면 소음이 납니다.
  #따라서 저속 천장은 0.35로 막고, 고속은 1.0으로 시원하게 엽니다.
  max_alpha = float(np.interp(v_ego_raw, [0.0, 5.0, 15.0], [0.25, 0.6, 1.0]))
  max_alpha = max(base_alpha, max_alpha)
  #5. 우아한 탄젠트(Tanh) 결합
  #오차에 속도 민감도를 곱한 값을 Tanh에 넣어 동적 가중치(0~1)를 구합니다.
  boost = math.tanh(error * sensitivity)
  #6. 최종 알파값 도출
  final_alpha = base_alpha + (max_alpha - base_alpha) * boost
  
  return (apply_angle * final_alpha) + (apply_angle_last * (1.0 - final_alpha))


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

    # For future parametrization / tuning
    self.angle_enable_smoothing_factor = True

    self._params = Params() if PARAMS_AVAILABLE else None
    if PARAMS_AVAILABLE:
      self.params.ANGLE_MIN_TORQUE_REDUCTION_GAIN = max(0.1, parse_tq_rdc_gain(
        self._params.get("HkgTuningAngleMinTorqueReductionGain")) or self.params.ANGLE_MIN_TORQUE_REDUCTION_GAIN)

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

    # steering torque
    if not self.CP.flags & HyundaiFlags.CANFD_ANGLE_STEERING:
      self.angle_limit_counter, apply_steer_req = common_fault_avoidance(abs(CS.out.steeringAngleDeg) >= MAX_ANGLE, CC.latActive,
                                                                         self.angle_limit_counter, MAX_ANGLE_FRAMES,
                                                                         MAX_ANGLE_CONSECUTIVE_FRAMES)
      new_torque = int(round(actuators.torque * self.params.STEER_MAX))
      apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.params)

    # angle control
    else:
      v_ego_raw = CS.out.vEgoRaw
      desired_angle = np.clip(actuators.steeringAngleDeg, -self.params.ANGLE_LIMITS.STEER_ANGLE_MAX, self.params.ANGLE_LIMITS.STEER_ANGLE_MAX)

      apply_angle = apply_steer_angle_limits_vm(desired_angle, self.apply_angle_last, v_ego_raw, CS.out.steeringAngleDeg, CC.latActive, self.params, self.VM)

      # if we are not the baseline model, we use the baseline model for further limits to prevent a panda block since it is hardcoded for baseline model.
      if  apply_angle is not None and self.CP.carFingerprint != ANGLE_SAFETY_BASELINE_MODEL:
        apply_angle = apply_steer_angle_limits_vm(apply_angle, self.apply_angle_last, v_ego_raw, CS.out.steeringAngleDeg, CC.latActive,
                                                  self.params, self.BASELINE_VM)                          

      # Use saturation-based torque reduction gain
      target_torque_reduction_gain = self.angle_torque_reduction_gain_controller.update(
        last_requested_angle=self.apply_angle_last,
        actual_angle=CS.out.steeringAngleDeg,
        lat_active=CC.latActive
      )

      # This method ensures that the torque gives up when overriding and controls the ramp rate to avoid feeling jittery.
      apply_torque = calculate_angle_torque_reduction_gain(self.params, CS, self.apply_torque_last, target_torque_reduction_gain)

      # apply_steer_req is True when we are actively attempting to steer and under the angle limit. Otherwise the user is overriding.
      apply_steer_req = CC.latActive and apply_torque != 0


      if apply_angle is not None and apply_steer_req and self.angle_enable_smoothing_factor:
        apply_angle = sp_smooth_angle(v_ego_raw, apply_angle, self.apply_angle_last)
      # Failsafe if we detected we'd violate safety
      if apply_angle is None:
        apply_torque = 0
        apply_angle = CS.out.steeringAngleDeg
        apply_steer_req = False


      # After we've used the last angle wherever we needed it, we now update it.
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
