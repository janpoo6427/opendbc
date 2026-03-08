import time


class TorqueReductionGainController:
    def __init__(self, angle_threshold=3.0, debounce_time=0.5, 
                 min_gain=0.0, max_gain=1.0, 
                 ramp_up_rate=0.1, ramp_down_rate=0.05):
        """
        [설정값]
        min_gain: 부스트 기능이 꺼져있을 때의 게인 (보통 0.0)
        max_gain: 부스트 기능이 최대로 켜졌을 때의 한계 게인 (보통 1.0)
        """
        self.angle_threshold = angle_threshold
        self.debounce_time = debounce_time
        
        # 부스트 게인의 범위 설정 (0.0 ~ 1.0)
        self.min_gain = min_gain
        self.max_gain = max_gain
        
        self.ramp_up_rate = ramp_up_rate
        self.ramp_down_rate = ramp_down_rate
        
        self.saturated_since = None
        self.current_boost = self.min_gain # 현재 부스트 량
        self.last_update_time = time.monotonic()

    def update(self, last_requested_angle, actual_angle, lat_active, current_active_torque):
        """
        current_active_torque: 속도에 따라 외부에서 계산된 기본 토크 게인 (0.15 ~ 0.6 등)
        """
        now = time.monotonic()
        dt = now - self.last_update_time
        self.last_update_time = now

        # 1. 오차 계산 및 포화 상태 확인
        angle_error = abs(last_requested_angle - actual_angle)
        saturated = lat_active and angle_error > self.angle_threshold

        # 2. 부스트 게인(Boost Gain) 계산 (0.0 -> 1.0)
        if saturated:
            if self.saturated_since is None:
                self.saturated_since = now
            elif (now - self.saturated_since) > self.debounce_time:
                # 오차가 지속되면 부스트를 올림
                self.current_boost = min(self.current_boost + self.ramp_up_rate * dt, self.max_gain)
        else:
            self.saturated_since = None
            # 오차가 해소되면 부스트를 내림
            self.current_boost = max(self.current_boost - self.ramp_down_rate * dt, self.min_gain)

        if not lat_active:
            self.current_boost = self.min_gain
            self.saturated_since = None

        # 3. [최종 결정] 기본 게인(Base) vs 부스트 게인(Boost) 중 큰 값 선택
        # - 정차 시 (Base 0.15, Boost 0.0) -> 0.15 (조용함)
        # - 정차 중 핸들 낌 (Base 0.15, Boost 0.8) -> 0.8 (탈출)
        # - 고속 주행 (Base 0.6, Boost 0.0) -> 0.6 (안정적)
        
        final_gain = max(current_active_torque, self.current_boost)
        
        return final_gain

    def reset(self):
        self.current_boost = self.min_gain
        self.saturated_since = None
        self.last_update_time = time.monotonic()