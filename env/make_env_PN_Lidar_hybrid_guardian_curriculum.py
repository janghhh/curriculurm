import gymnasium as gym
import numpy as np
import airsim
import math
import time
import random
from collections import deque


class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",
        follower_names=("Follower0", "Follower1", "Follower2"),
        port=41451,
        step_length=0.01,
        leader_velocity=3.0,
        optimal_distance=10.0,
        far_cutoff=60.0,
        too_close=0.5,
        dt=0.15,
        do_visualize=True
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]
        print("[DEBUG env file] __init__ called")
        print("[DEBUG env follower_names]", follower_names)
        print("[DEBUG env possible_agents]", list(follower_names))

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 1.5
        self.STOP_DISTANCE_LEADER_OBSTACLE = 1.0

        # 속도/액션 버퍼
        self.vmax_self = 2.0
        self._timestep = float(dt)

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = -10.0
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)
        self.max_cmd_speed = self.step_length / self.dt
        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        # Lidar 메모리: { "AgentName": { "TargetName": [dx, dy] } }
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        # Lidar 센서 이름 매핑
        self.lidar_names = {
            agent: f"{agent}_LidarSensor" for agent in self.possible_agents
        }

        # States
        self.step_count = 0
        self.episode_count = 0

        # 통계 계산을 위한 Deque
        self.stats_history = {
            "win": deque(maxlen=20),
            "coll_leader": deque(maxlen=20),
            "coll_drone": deque(maxlen=20),
            "coll_obj": deque(maxlen=20)
        }

        # ===== 동적 장애물 2개 설정 =====
        self.dynamic_names = ["DynamicObstacle0", "DynamicObstacle1"]
        self.num_enemies = len(self.dynamic_names)
        # 제거(전장에서 이탈) 상태 추적
        self.eliminated_enemies = set()   # 요격 성공으로 제거된 적기 이름
        self.eliminated_agents = set()    # 요격 시 함께 제거된 아군 이름

        # 각 적기별 FSM 상태
        self._obstacle_states = {}    # "IDLE" / "ATTACK" / "ELIMINATED"
        self._obs_step_timers = {}
        self._idle_wait_steps = {}
        self._obstacle_speeds = {}

        # 각 적기별 interceptor 매핑
        self.interceptor_map = {}     # { "DynamicObstacle0": "Follower1", ... }
        self.role_map = {a: "escort" for a in self.possible_agents}

        # 동적 장애물 관련 (하위호환용)
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}

        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1
        self.K_enemy = self.num_enemies  # 적기 2대
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy

        # 관측 공간 범위 정의
        low_rel_pos = -1.0
        high_rel_pos = 1.0
        low_vel = -1.0
        high_vel = 1.0
        low_rate = -1.0
        high_rate = 1.0
        low_self_state = -1.0
        high_self_state = 1.0

        # [리더(2)] + [아군(2)*K_ally] + [적(4)*K_enemy] + [self_state(2)]
        per_agent_low = (
            [low_rel_pos, low_rel_pos] +                          # Leader (dx, dy)
            [low_rel_pos, low_rel_pos] * self.num_ally +          # Allies (dx, dy)
            [low_rel_pos, low_rel_pos, low_vel, low_rate] * self.num_enemy +  # Enemies
            [low_self_state] * 2                                  # Self (vx, vy)
        )
        per_agent_high = (
            [high_rel_pos, high_rel_pos] +
            [high_rel_pos, high_rel_pos] * self.num_ally +
            [high_rel_pos, high_rel_pos, high_vel, high_rate] * self.num_enemy +
            [high_self_state] * 2
        )

        obs_dim = len(per_agent_low)
        share_obs_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.MAX_YAW = math.radians(90)
        self.MAX_SPEED = 10

        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float64,
            )
            for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,),
            dtype=np.float32,
        )

        # PN 보상용 버퍼들
        self._prev_d_leader_enemy = {dn: None for dn in self.dynamic_names}
        self._prev_d_agent_enemy = {
            a: {dn: None for dn in self.dynamic_names}
            for a in self.possible_agents
        }
        self._prev_los_angle = {
            a: {dn: None for dn in self.dynamic_names}
            for a in self.possible_agents
        }

        # =========================
        # Simple curriculum rewards
        # =========================
        self.REWARD_STEP = -0.001
        self.REWARD_SUCCESS_SINGLE = 0.5     # 적기 1대 요격 시 중간 보상
        self.REWARD_SUCCESS_ALL = 1.0        # 적기 전부 요격 시 최종 보상
        self.REWARD_SUCCESS_ESCORT_INTERCEPT = 0.3  # 호위 드론이 요격한 경우
        self.REWARD_FAIL = -1.0

        # 요격 성공 판정 반경
        self.hit_radius_start = 15.0
        self.hit_radius = 15.0
        self.min_hit_radius = 1.5
        self.hit_radius_decay = 0.1
        self.successes_per_level = 5
        self.total_successes = 0

        # 호위 에이전트 보상
        self.ESCORT_MAX_RADIUS = 30.0
        self.ESCORT_OUTSIDE_PENALTY = -0.05

        self.current_episode_hit_radius = self.hit_radius

        self.REAL_LEADER_SPEED = 1.5
        self.leader_move_timer = 0
        self.current_leader_vel = [0.0, 0.0]

        # 위치 캐시
        self.start_location = {}
        self.current_location = {}

        self.obs_speed_min = 3.0
        self.obs_speed_max = 3.0

        # 시각화 설정
        self.VIZ_REFRESH_SEC = 0.1
        self.INTERCEPTOR_RING_RADIUS = 3.0
        self.INTERCEPTOR_MARK_HEIGHT = 6.0

        # 평가 지표(IR, SIR) 계산용 변수
        self.MISS_THRESHOLD = 5.0
        self.primary_miss_occurred = False
        self._prev_dist_to_enemy = {
            a: {dn: 999.0 for dn in self.dynamic_names}
            for a in self.possible_agents
        }

        # 에피소드 내 요격 성공 카운트
        self.enemies_killed_this_episode = 0
        self.kill_log = []

        # 클라이언트 셋업
        self.client = airsim.MultirotorClient(ip=ip_address, port=port)
        self.client.confirmConnection()

        self._last_visualize_t = time.time()

    # ======================================================================
    # Lidar 관련
    # ======================================================================
    def _get_lidar_measurement(self, agent_name, target_name):
        """
        Lidar로 타겟을 관측합니다.
        제거된 개체는 관측하지 않고 기본값(0,0) 반환.
        """
        # 제거된 개체는 관측 불가
        if target_name in self.eliminated_enemies or target_name in self.eliminated_agents:
            return 0.0, 0.0, False

        lidar_name = self.lidar_names[agent_name]
        lidar_data = self.client.getLidarData(lidar_name, vehicle_name=agent_name)

        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        detected = False

        if len(points) > 2:
            my_pos = self.current_location[agent_name].position
            tgt_pos = self.current_location[target_name].position

            gt_dx = tgt_pos.x_val - my_pos.x_val
            gt_dy = tgt_pos.y_val - my_pos.y_val

            dist_sq = (points[:, 0] - gt_dx)**2 + (points[:, 1] - gt_dy)**2
            mask = dist_sq < (5.0)**2
            target_points = points[mask]

            if len(target_points) > 0:
                mean_pos = np.mean(target_points, axis=0)
                raw_dx = mean_pos[0]
                raw_dy = mean_pos[1]

                norm_dx = np.clip(raw_dx / 100.0, -1.0, 1.0)
                norm_dy = np.clip(raw_dy / 100.0, -1.0, 1.0)

                self.lidar_memory[agent_name][target_name] = [float(norm_dx), float(norm_dy)]
                detected = True

        final_val = self.lidar_memory[agent_name].get(target_name, [0.0, 0.0])

        return final_val[0], final_val[1], detected

    def _calculate_lidar_dynamics(self, agent, target, curr_x, curr_y):
        """Lidar의 (현재 위치 - 이전 위치) 차분으로 속도 정보 계산"""
        if agent not in self.prev_lidar_pos:
            self.prev_lidar_pos[agent] = {}
        if target not in self.prev_lidar_pos[agent]:
            self.prev_lidar_pos[agent][target] = [curr_x, curr_y]

        prev_x, prev_y = self.prev_lidar_pos[agent][target]
        dt = self.dt if self.dt > 1e-6 else 0.01

        vx_rel = (curr_x - prev_x) / dt
        vy_rel = (curr_y - prev_y) / dt

        R_vec = np.array([curr_x, curr_y])
        V_vec = np.array([vx_rel, vy_rel])
        dist = np.linalg.norm(R_vec) + 1e-6

        closing_speed = -float(np.dot(R_vec, V_vec)) / dist
        cross_prod = float(R_vec[0]*V_vec[1] - R_vec[1]*V_vec[0])
        los_rate = cross_prod / (dist**2)

        norm_closing = np.clip(closing_speed / 30.0, -1.0, 1.0)
        norm_los = np.clip(los_rate / 10.0, -1.0, 1.0)

        return norm_closing, norm_los

    def _get_current_location(self):
        self.current_location = {}
        self.current_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            if agent not in self.eliminated_agents:
                self.current_location[agent] = self.client.simGetObjectPose(agent)
        for dn in self.dynamic_names:
            if dn not in self.eliminated_enemies:
                self.current_location[dn] = self.client.simGetObjectPose(dn)

    # ======================================================================
    # 초기화/이동/시각화 관련
    # ======================================================================
    def _hover(self, name):
        self.client.moveByVelocityZAsync(
            vx=0.0, vy=0.0,
            z=self.fixed_z,
            duration=0.3,
            vehicle_name=name
        ).join()

        try:
            self.client.hoverAsync(vehicle_name=name).join()
        except:
            pass

    def _setup_flight(self):
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
            self.client.armDisarm(True, vehicle_name=agent)

        for dn in self.dynamic_names:
            self.client.enableApiControl(True, vehicle_name=dn)
            self.client.armDisarm(True, vehicle_name=dn)

        # 1. 이륙
        cmds = []
        cmds.append(self.client.takeoffAsync(vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.takeoffAsync(vehicle_name=agent))
        for dn in self.dynamic_names:
            cmds.append(self.client.takeoffAsync(vehicle_name=dn))
        for c in cmds:
            c.join()

        # 2. 텔레포트
        pose_leader = airsim.Pose(
            airsim.Vector3r(0, 0, self.fixed_z),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        self.client.simSetVehiclePose(pose_leader, True, vehicle_name="Drone1")

        for i, agent in enumerate(self.possible_agents):
            pose_agent = airsim.Pose(
                airsim.Vector3r(0, 0, self.fixed_z),
                airsim.Quaternionr(0, 0, 0, 1)
            )
            self.client.simSetVehiclePose(pose_agent, True, vehicle_name=agent)

        for dn in self.dynamic_names:
            pose_enemy = airsim.Pose(
                airsim.Vector3r(0, 0, self.fixed_z),
                airsim.Quaternionr(0, 0, 0, 1)
            )
            self.client.simSetVehiclePose(pose_enemy, True, vehicle_name=dn)

        # 3. 속도 0으로 강제 초기화
        cmds = []
        cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=agent))
        for dn in self.dynamic_names:
            cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=dn))
        for c in cmds:
            c.join()

        time.sleep(0.5)

        # 4. 초기 위치 저장
        self.start_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        for dn in self.dynamic_names:
            self.start_location[dn] = self.client.simGetObjectPose(dn)

        # Hover
        self._hover("Drone1")
        for agent in self.possible_agents:
            self._hover(agent)
        for dn in self.dynamic_names:
            self._hover(dn)

    def _update_leader_movement(self):
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        if self.leader_move_timer <= 0:
            pos = self.client.simGetObjectPose("Drone1").position
            dist_from_origin = math.hypot(pos.x_val, pos.y_val)

            if dist_from_origin > 50.0:
                angle = math.atan2(-pos.y_val, -pos.x_val)
                angle += random.uniform(-0.3, 0.3)
            else:
                angle = random.uniform(0, 2 * math.pi)

            vx = math.cos(angle) * self.REAL_LEADER_SPEED
            vy = math.sin(angle) * self.REAL_LEADER_SPEED
            self.current_leader_vel = [vx, vy]
            self.leader_move_timer = random.randint(20, 50)

        self.leader_move_timer -= 1

        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0.0)
        self.client.moveByVelocityZAsync(
            vx=self.current_leader_vel[0],
            vy=self.current_leader_vel[1],
            z=self.fixed_z,
            duration=self.dt,
            yaw_mode=yaw_mode,
            vehicle_name="Drone1"
        )

    def _visualize_circles(self):
        try:
            leader_pose = self.client.simGetObjectPose("Drone1")
            leader_pos = leader_pose.position
            center = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val], dtype=float)

            def ring_points(radius, n=72, z=None):
                if z is None:
                    z = center[2]
                pts = []
                for i in range(n + 1):
                    ang = (i / n) * 2 * np.pi
                    x = center[0] + radius * np.cos(ang)
                    y = center[1] + radius * np.sin(ang)
                    pts.append(airsim.Vector3r(x, y, z))
                return pts

            def make_ring(cx, cy, cz, radius, n=40):
                pts = []
                for i in range(n + 1):
                    ang = (i / n) * 2 * np.pi
                    x = cx + radius * np.cos(ang)
                    y = cy + radius * np.sin(ang)
                    pts.append(airsim.Vector3r(x, y, cz))
                return pts

            line_thickness = 15.0

            # 30m 원 (리더 기준)
            self.client.simPlotLineStrip(
                ring_points(30.0),
                [1.0, 1.0, 0.0, 0.95],
                thickness=line_thickness,
                duration=self.VIZ_REFRESH_SEC + 0.05,
                is_persistent=True
            )

            # 60m 원 (리더 기준)
            self.client.simPlotLineStrip(
                ring_points(60.0),
                [0.0, 1.0, 0.0, 0.95],
                thickness=line_thickness,
                duration=self.VIZ_REFRESH_SEC + 0.05,
                is_persistent=True
            )

            # ===== 적기-요격기 쌍별 색상 매칭 시각화 =====
            pair_colors = [
                [1.0, 0.2, 0.2, 1.0],   # 빨강 (DynamicObstacle0)
                [0.2, 0.5, 1.0, 1.0],   # 파랑 (DynamicObstacle1)
                [1.0, 0.5, 0.0, 1.0],   # 주황 (확장용)
                [0.8, 0.2, 1.0, 1.0],   # 보라 (확장용)
            ]

            for idx, dn in enumerate(self.dynamic_names):
                if dn in self.eliminated_enemies:
                    continue

                color = pair_colors[idx % len(pair_colors)]

                # --- 적기 링 ---
                enemy_pose = self.client.simGetObjectPose(dn)
                ep = enemy_pose.position
                ex, ey, ez = ep.x_val, ep.y_val, ep.z_val

                self.client.simPlotLineStrip(
                    make_ring(ex, ey, ez, self.INTERCEPTOR_RING_RADIUS + 1.0),
                    color,
                    thickness=12.0,
                    duration=self.VIZ_REFRESH_SEC + 0.05,
                    is_persistent=True
                )

                # --- 매칭된 요격기 링 + 연결선 ---
                intr_name = self.interceptor_map.get(dn)
                if intr_name is None or intr_name in self.eliminated_agents:
                    continue

                intr_pose = self.client.simGetObjectPose(intr_name)
                ip = intr_pose.position
                ix, iy, iz = ip.x_val, ip.y_val, ip.z_val

                # 요격기 링 (같은 색상)
                self.client.simPlotLineStrip(
                    make_ring(ix, iy, iz, self.INTERCEPTOR_RING_RADIUS),
                    color,
                    thickness=10.0,
                    duration=self.VIZ_REFRESH_SEC + 0.05,
                    is_persistent=True
                )

                # 요격기 → 적기 연결선 (반투명)
                self.client.simPlotLineStrip(
                    [airsim.Vector3r(ix, iy, iz), airsim.Vector3r(ex, ey, ez)],
                    [color[0], color[1], color[2], 0.5],
                    thickness=5.0,
                    duration=self.VIZ_REFRESH_SEC + 0.05,
                    is_persistent=True
                )

            # ===== 미배정 활성 적기: 회색 링 =====
            for dn in self.dynamic_names:
                if dn in self.eliminated_enemies:
                    continue
                if dn not in self.interceptor_map or self.interceptor_map.get(dn) is None:
                    enemy_pose = self.client.simGetObjectPose(dn)
                    ep = enemy_pose.position
                    self.client.simPlotLineStrip(
                        make_ring(ep.x_val, ep.y_val, ep.z_val, self.INTERCEPTOR_RING_RADIUS + 1.0),
                        [0.6, 0.6, 0.6, 0.8],
                        thickness=8.0,
                        duration=self.VIZ_REFRESH_SEC + 0.05,
                        is_persistent=True
                    )

        except Exception as e:
            print(f"[Visualization Error] {e}")

    # ======================================================================
    # 보상/종료 관련
    # ======================================================================
    def _compute_reward(self, agent):
        """에이전트가 제거되었으면 보상 0"""
        if agent in self.eliminated_agents:
            return 0.0

        role = self.role_map.get(agent, "escort")

        if role == "interceptor":
            return float(self.REWARD_STEP)

        return self._compute_escort_reward(agent)

    def _compute_escort_reward(self, agent):
        if agent in self.eliminated_agents:
            return 0.0

        pa = self.current_location[agent].position
        pl = self.current_location["Drone1"].position

        d_leader = math.hypot(pa.x_val - pl.x_val, pa.y_val - pl.y_val)

        if d_leader > self.ESCORT_MAX_RADIUS:
            return float(self.ESCORT_OUTSIDE_PENALTY)

        return 0.0

    def _build_intercept_reward(self, hit_agent, enemy_name):
        """
        적기 1대 요격 성공 시 중간 보상.
        - interceptor가 잡으면 0.5
        - escort가 잡으면 0.3
        """
        if self.role_map.get(hit_agent, "escort") == "interceptor":
            r = float(self.REWARD_SUCCESS_SINGLE)
        else:
            r = float(self.REWARD_SUCCESS_ESCORT_INTERCEPT)

        return {a: r for a in self.possible_agents}

    def _build_all_clear_reward(self):
        """적기 전부 요격 → 최종 보상 1.0"""
        return {a: float(self.REWARD_SUCCESS_ALL) for a in self.possible_agents}

    def _update_curriculum_on_success(self):
        prev_radius = self.hit_radius
        self.total_successes += 1

        level = self.total_successes // self.successes_per_level
        self.hit_radius = max(
            self.min_hit_radius,
            self.hit_radius_start - level * self.hit_radius_decay
        )

        if self.hit_radius < prev_radius:
            print(
                f"[Curriculum] total_successes={self.total_successes} | "
                f"hit_radius: {prev_radius:.2f} -> {self.hit_radius:.2f} m"
            )

    def _eliminate_pair(self, agent_name, enemy_name):
        """
        아군-적기 쌍을 전장에서 제거한다.
        제거된 개체는 먼 곳으로 텔레포트하고 정지시킨다.
        """
        self.kill_log.append({
            "agent": agent_name,
            "enemy": enemy_name,
            "role": self.role_map.get(agent_name, "escort")
        })

        self.eliminated_agents.add(agent_name)
        self.eliminated_enemies.add(enemy_name)
        self._obstacle_states[enemy_name] = "ELIMINATED"

        # 역할 초기화
        self.role_map[agent_name] = "eliminated"

        # interceptor_map에서 해당 적기 제거
        if enemy_name in self.interceptor_map:
            del self.interceptor_map[enemy_name]

        # 다른 적기의 interceptor가 이 아군이었으면 재배정 필요
        for dn in self.dynamic_names:
            if dn in self.eliminated_enemies:
                continue
            if self.interceptor_map.get(dn) == agent_name:
                self.interceptor_map[dn] = None
                self._assign_interceptor_for_enemy(dn)

        # 물리적으로 먼 곳에 격리
        far_pose = airsim.Pose(
            airsim.Vector3r(9999, 9999, -100),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        self.client.simSetVehiclePose(far_pose, True, vehicle_name=agent_name)
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=agent_name)

        self.client.simSetVehiclePose(far_pose, True, vehicle_name=enemy_name)
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=enemy_name)

        print(f"[Eliminate] {agent_name} & {enemy_name} removed from battlefield")

    def _get_active_agents(self):
        """제거되지 않은 활성 아군 목록"""
        return [a for a in self.possible_agents if a not in self.eliminated_agents]

    def _get_active_enemies(self):
        """제거되지 않은 활성 적기 목록"""
        return [dn for dn in self.dynamic_names if dn not in self.eliminated_enemies]
    
    def _calc_interceptor_kill(self, status):
        """interceptor가 요격한 비율. 실패 에피소드면 0."""
        if "SUCCESS" not in status:
            return 0.0
        if len(self.kill_log) == 0:
            return 0.0
        interceptor_kills = sum(1 for k in self.kill_log if k["role"] == "interceptor")
        return interceptor_kills / len(self.kill_log)

    def _end_episode(self, reward, status):
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        if isinstance(reward, dict):
            reward_dict = reward
        else:
            reward_dict = {a: float(reward) for a in self.possible_agents}

        is_success = 1.0 if status == "SUCCESS_ALL_ENEMIES_ELIMINATED" else 0.0
        is_partial = 1.0 if "SUCCESS_DISTANCE" in status and "ALL" not in status else 0.0
        is_leader_hit = 1.0 if "LEADER_AND_DYNAMIC" in status else 0.0

        if "AGENT_AGENT" in status or "AGENT_LEADER" in status:
            is_ally_collision = 1.0
        else:
            is_ally_collision = 0.0

        is_obj_collision = 1.0 if "FAR_CUTOFF" in status else 0.0

        intercept_range = 0.0
        if is_success or is_partial:
            lx = self.current_location["Drone1"].position.x_val
            ly = self.current_location["Drone1"].position.y_val
            # 가장 마지막으로 제거된 적기 기준 (활성 적이 없으면 0)
            active_enemies = self._get_active_enemies()
            if len(active_enemies) > 0:
                last_e = active_enemies[0]
                ex = self.current_location[last_e].position.x_val
                ey = self.current_location[last_e].position.y_val
                intercept_range = math.hypot(lx - ex, ly - ey)

        self.stats_history["win"].append(is_success)
        self.stats_history["coll_leader"].append(is_leader_hit)
        self.stats_history["coll_drone"].append(is_ally_collision)
        self.stats_history["coll_obj"].append(is_obj_collision)

        def get_rate(key):
            if len(self.stats_history[key]) > 0:
                return sum(self.stats_history[key]) / len(self.stats_history[key])
            return 0.0

        win_rate = get_rate("win")

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(float(reward_dict.get(agent, 0.0)))
            _terminations_list.append(True)

            _infos_list.append({
                agent: {
                    "final_status": status,
                    "episode_success": is_success,
                    "episode_leader_hit": is_leader_hit,
                    "episode_ally_collision": is_ally_collision,
                    "win_rate": win_rate,
                    "cur_episode_steps": self.step_count,
                    "episode_intercept_range": intercept_range,
                    "primary_miss_occurred": self.primary_miss_occurred,
                    "episode_hit_radius": self.current_episode_hit_radius,
                    "next_hit_radius": self.hit_radius,
                    "total_successes": self.total_successes,
                    "role": self.role_map.get(agent, "escort"),
                    "enemies_killed": self.enemies_killed_this_episode,
                    "eliminated_agents": list(self.eliminated_agents),
                    "eliminated_enemies": list(self.eliminated_enemies),
                    "interceptor_kill": self._calc_interceptor_kill(status),
                }
            })

        print(
            f"[{self.episode_count} Ep] WinRate: {win_rate:.2f} | Status: {status} | "
            f"enemies_killed={self.enemies_killed_this_episode}/{self.num_enemies} | "
            f"episode_hit_radius={self.current_episode_hit_radius:.2f} | "
            f"next_hit_radius={self.hit_radius:.2f} | total_successes={self.total_successes}"
        )
        return _obs_list, _rewards_list, _terminations_list, _infos_list

    # --------------------- 동적장애물 FSM (2개) ---------------------
    def _update_dynamic_obstacles(self):
        """각 적기별로 독립적인 FSM 업데이트"""
        for dn in self.dynamic_names:
            if dn in self.eliminated_enemies:
                continue
            self._update_single_obstacle(dn)

    def _update_single_obstacle(self, dn):
        self._obs_step_timers[dn] += 1
        target_speed = self._obstacle_speeds.get(dn, 3.0)
        state = self._obstacle_states[dn]

        if state == "IDLE":
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=dn)
            if self._obs_step_timers[dn] >= self._idle_wait_steps[dn]:
                self._obstacle_states[dn] = "ATTACK"
                self._obs_step_timers[dn] = 0
                print(f"[{dn}] IDLE -> ATTACK")

        elif state == "ATTACK":
            try:
                leader_pose_live = self.client.simGetObjectPose("Drone1").position
                obs_pos_live = self.client.simGetObjectPose(dn).position

                l_vec = np.array([leader_pose_live.x_val, leader_pose_live.y_val, leader_pose_live.z_val])
                o_vec = np.array([obs_pos_live.x_val, obs_pos_live.y_val, obs_pos_live.z_val])

                diff = l_vec - o_vec
                dist = np.linalg.norm(diff)

                if dist > 0.5:
                    direction = diff / dist
                    vel = direction * target_speed

                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                        duration=0.1,
                        vehicle_name=dn
                    )
            except Exception as e:
                print(f"[{dn}] Attack Logic Error: {e}")

        # "ELIMINATED" 상태는 아무것도 하지 않음

    def _teleport_obstacle_randomly(self, dn):
        leader_pos = self.client.simGetVehiclePose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        radius = 55.0
        angle = random.uniform(0, 2 * math.pi)

        tx = lx - 20 + radius * math.cos(angle)
        ty = ly - 20 + radius * math.sin(angle)
        tz = self.fixed_z

        pose = airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose, True, vehicle_name=dn)
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=dn).join()

    def _reset_obstacle_logic(self):
        """모든 적기를 초기화"""
        self.eliminated_enemies.clear()
        self.eliminated_agents.clear()
        self.enemies_killed_this_episode = 0
        self.kill_log = []

        for idx, dn in enumerate(self.dynamic_names):
            self._teleport_obstacle_randomly(dn)
            self._obstacle_states[dn] = "IDLE"
            self._obs_step_timers[dn] = 0
            # 각 적기마다 다른 대기 시간 부여 → 시차 공격
            base_wait = random.randint(10, 30)
            self._idle_wait_steps[dn] = base_wait + idx * random.randint(15, 40)
            self._obstacle_speeds[dn] = random.uniform(self.obs_speed_min, self.obs_speed_max)
            print(f"[{dn}] Reset to IDLE. Waiting {self._idle_wait_steps[dn]} steps.")

        self.interceptor_map = {}
        self.role_map = {a: "escort" for a in self.possible_agents}

    def _assign_interceptor_for_enemy(self, enemy_name):
        """
        특정 적기에 대해 interceptor를 배정한다.
        - 이미 다른 적기의 interceptor인 아군은 제외
        - 제거된 아군은 제외
        """
        if enemy_name in self.eliminated_enemies:
            return False

        # 현재 다른 적기의 interceptor로 배정된 아군 집합
        already_assigned = set()
        for dn, intr in self.interceptor_map.items():
            if intr is not None and dn != enemy_name:
                already_assigned.add(intr)

        candidates = [
            a for a in self.possible_agents
            if a not in self.eliminated_agents and a not in already_assigned
        ]

        if len(candidates) == 0:
            print(f"[Role] No available agent for {enemy_name}")
            if enemy_name in self.interceptor_map:
                del self.interceptor_map[enemy_name]
            return False

        # Lidar로 적을 볼 수 있는 후보 필터링
        visible_dist = {}
        for agent in candidates:
            ex, ey, detected = self._get_lidar_measurement(agent, enemy_name)
            if detected:
                dist = math.hypot(ex * 100.0, ey * 100.0)
                visible_dist[agent] = dist

        if len(visible_dist) == 0:
            # 아무도 못 봤으면 가장 가까운 아군(GT 기반)으로 fallback
            fallback_dist = {}
            for agent in candidates:
                if agent in self.current_location and enemy_name in self.current_location:
                    pa = self.current_location[agent].position
                    pe = self.current_location[enemy_name].position
                    d = math.hypot(pa.x_val - pe.x_val, pa.y_val - pe.y_val)
                    fallback_dist[agent] = d
            if len(fallback_dist) > 0:
                chosen = min(fallback_dist, key=fallback_dist.get)
            else:
                chosen = candidates[0]
            print(f"[Role] No LiDAR detection for {enemy_name}, fallback assign: {chosen}")
        else:
            chosen = min(visible_dist, key=visible_dist.get)

        self.interceptor_map[enemy_name] = chosen
        self.role_map[chosen] = "interceptor"

        # 나머지 아군 중 어떤 적의 interceptor도 아닌 애들은 escort
        all_interceptors = set(self.interceptor_map.values())
        for a in self.possible_agents:
            if a in self.eliminated_agents:
                self.role_map[a] = "eliminated"
            elif a not in all_interceptors:
                self.role_map[a] = "escort"

        print(f"[Role-LiDAR] {enemy_name} -> interceptor={chosen}")
        return True

    def _check_distance_collision(self, name_a, name_b, threshold):
        """제거된 개체 간 거리는 항상 충돌하지 않은 것으로 처리"""
        if name_a in self.eliminated_agents or name_a in self.eliminated_enemies:
            return False, 9999.0
        if name_b in self.eliminated_agents or name_b in self.eliminated_enemies:
            return False, 9999.0

        pa = self.current_location.get(name_a)
        pb = self.current_location.get(name_b)
        if pa is None or pb is None:
            return False, 9999.0

        pa = pa.position
        pb = pb.position

        dx = pa.x_val - pb.x_val
        dy = pa.y_val - pb.y_val
        dist = math.sqrt(dx * dx + dy * dy)
        return dist < threshold, dist

    # ======================================================================
    # RL/PettingZoo API
    # ======================================================================
    @property
    def observation_space(self):
        return [self.observation_spaces[a] for a in self.possible_agents]

    @property
    def action_space(self):
        return [self.action_spaces[a] for a in self.possible_agents]

    @property
    def share_observation_space(self):
        return [self.share_observation_spaces for _ in self.possible_agents]

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _get_obs(self, agent):
        """
        제거된 아군은 전부 0인 관측을 반환한다.
        제거된 적기는 (0, 0, 0, 0)으로 채운다.
        """
        obs_dim = self.observation_spaces[self.possible_agents[0]].shape[0]

        if agent in self.eliminated_agents:
            return np.zeros(obs_dim, dtype=np.float32)

        # 1. 내 속도
        try:
            my_state = self.client.getMultirotorState(vehicle_name=agent)
            vx = my_state.kinematics_estimated.linear_velocity.x_val
            vy = my_state.kinematics_estimated.linear_velocity.y_val
            norm_vx = np.clip(vx / self.MAX_SPEED, -1.0, 1.0)
            norm_vy = np.clip(vy / self.MAX_SPEED, -1.0, 1.0)
        except:
            norm_vx, norm_vy = 0.0, 0.0

        # 2. 리더 위치 (Lidar)
        lx, ly, _ = self._get_lidar_measurement(agent, "Drone1")
        _leader_feats = [lx, ly]

        # 3. 아군 위치 (Lidar)
        _ally_feats = []
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            if other in self.eliminated_agents:
                _ally_feats.extend([0.0, 0.0])
            else:
                ox, oy, _ = self._get_lidar_measurement(agent, other)
                _ally_feats.append(ox)
                _ally_feats.append(oy)

        # 4. 적기들 (순서 고정: dynamic_names 순서대로)
        _dynamic_feats = []
        for dn in self.dynamic_names:
            if dn in self.eliminated_enemies:
                _dynamic_feats.extend([0.0, 0.0, 0.0, 0.0])
            else:
                ex, ey, is_enemy_visible = self._get_lidar_measurement(agent, dn)

                ex_meter = ex * 100.0
                ey_meter = ey * 100.0

                if is_enemy_visible:
                    closing_norm, los_norm = self._calculate_lidar_dynamics(
                        agent, dn, ex_meter, ey_meter
                    )
                else:
                    closing_norm, los_norm = 0.0, 0.0

                _dynamic_feats.extend([ex, ey, closing_norm, los_norm])

        # 5. 최종 결합
        obs = np.concatenate([
            np.array(_leader_feats, dtype=np.float32).flatten(),
            np.array(_ally_feats, dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
            np.array([norm_vx, norm_vy], dtype=np.float32)
        ]).astype(np.float32)

        return obs

    def _do_action(self, actions):
        actions = np.clip(actions, -1, 1)
        dt = self.dt

        for i, agent in enumerate(self.possible_agents):
            if agent in self.eliminated_agents:
                continue

            a = actions[i]
            v_forward = float(a[0]) * self.MAX_SPEED
            v_lateral = float(a[1]) * self.MAX_SPEED

            sp = math.hypot(v_forward, v_lateral)
            if sp > self.MAX_SPEED:
                s = self.MAX_SPEED / (sp + 1e-6)
                v_forward *= s
                v_lateral *= s

            vx = v_forward
            vy = v_lateral

            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0.0)

            self.client.moveByVelocityZAsync(
                vx=vx, vy=vy, z=self.fixed_z, duration=dt,
                yaw_mode=yaw_mode,
                vehicle_name=agent
            )

    def _get_rewards(self, per_agent_results):
        return [float(r) for r in per_agent_results]

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        print(
            f"에피소드: {self.episode_count} | 소비한 스텝 수: {self.step_count} | "
            f"total_successes: {self.total_successes} | "
            f"hit_radius: {self.hit_radius:.2f} m"
        )

        self.step_count = 0
        self.agents = self.possible_agents[:]

        self.leader_move_timer = 0
        self.current_leader_vel = [0.0, 0.0]

        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        self._reset_obstacle_logic()
        self._get_current_location()

        self._prev_d_agent_enemy = {
            a: {dn: None for dn in self.dynamic_names}
            for a in self.possible_agents
        }
        self._last_pose.clear()
        self._last_time.clear()

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

        # 메모리 및 이전 위치 초기화
        self.lidar_memory = {}
        self.prev_lidar_pos = {}

        all_targets = ["Drone1"] + [a for a in self.possible_agents] + self.dynamic_names

        for agent in self.possible_agents:
            self.lidar_memory[agent] = {}
            self.prev_lidar_pos[agent] = {}

            default_real_pos = [100.0, 100.0]

            for tgt in all_targets:
                if tgt == agent:
                    continue
                self.lidar_memory[agent][tgt] = [1.0, 1.0]
                self.prev_lidar_pos[agent][tgt] = default_real_pos

        self.primary_miss_occurred = False
        self.current_episode_hit_radius = self.hit_radius
        self._prev_dist_to_enemy = {
            a: {dn: 999.0 for dn in self.dynamic_names}
            for a in self.possible_agents
        }

        obs_list = [self._get_obs(a) for a in self.agents]
        return obs_list

    def step(self, actions):
        self.step_count += 1
        self._pending_mid_reward = None

        if self.step_count >= 1000:
            print(f"⏳[시간 초과] 스텝 {self.step_count} 도달! → 실패 처리")
            return self._end_episode(self.REWARD_FAIL, "FAIL_TIMEOUT_LEADER_HIT")

        per_agent_obs, per_agent_results, per_agent_infos = [], [], []

        self._do_action(actions)
        # self._update_leader_movement()
        self._update_dynamic_obstacles()
        self._get_current_location()

        # 공격 중인 적기에 대해 interceptor가 없으면 배정
        for dn in self.dynamic_names:
            if dn in self.eliminated_enemies:
                continue
            if self._obstacle_states[dn] == "ATTACK":
                if self.interceptor_map.get(dn) is None or \
                   self.interceptor_map.get(dn) in self.eliminated_agents:
                    self._assign_interceptor_for_enemy(dn)

        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= self.VIZ_REFRESH_SEC:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        # ---- 에이전트별 체크 ----
        active_agents = self._get_active_agents()

        for agent in active_agents:
            # === 적기별 거리 체크 (miss 판정 & 요격 성공) ===
            for dn in self._get_active_enemies():
                ex = self.current_location[dn].position.x_val
                ey = self.current_location[dn].position.y_val
                ax = self.current_location[agent].position.x_val
                ay = self.current_location[agent].position.y_val

                curr_dist = math.hypot(ex - ax, ey - ay)
                prev_dist = self._prev_dist_to_enemy[agent].get(dn, 999.0)

                # miss 판정
                if (
                    curr_dist > prev_dist
                    and prev_dist < self.MISS_THRESHOLD
                    and prev_dist > self.COLLISION_THRESHOLD
                ):
                    self.primary_miss_occurred = True
                    if self._obstacle_states[dn] == "ATTACK" and \
                       self.interceptor_map.get(dn) == agent:
                        print(f"[Handoff] {agent} missed {dn} -> reassign")
                        self.interceptor_map[dn] = None
                        self._assign_interceptor_for_enemy(dn)

                self._prev_dist_to_enemy[agent][dn] = curr_dist

                # 요격 성공 체크
                if self._obstacle_states[dn] == "ATTACK":
                    hit, d = self._check_distance_collision(
                        agent, dn, threshold=self.current_episode_hit_radius
                    )
                    if hit:
                        role = self.role_map.get(agent, "escort")
                        print(
                            f"🎯[요격 성공] {agent}({role}) -> {dn} | "
                            f"hit_radius={self.current_episode_hit_radius:.2f}m"
                        )

                        # 중간 보상 계산
                        mid_reward = self._build_intercept_reward(agent, dn)

                        # 쌍 제거
                        self._eliminate_pair(agent, dn)
                        self.enemies_killed_this_episode += 1
                        self._update_curriculum_on_success()

                        if len(self._get_active_enemies()) == 0:
                            print(f"🏆[전체 요격 성공] 모든 적기 제거!")
                            # 기본 step 보상 + 킬 보상 합산 (중간 킬과 동일 방식)
                            combined = {}
                            for a in self.possible_agents:
                                base = self._compute_reward(a)
                                combined[a] = base + mid_reward.get(a, 0.0)
                            return self._end_episode(combined, "SUCCESS_ALL_ENEMIES_ELIMINATED")

                        # 아직 적이 남아있으면 중간 보상만 지급하고 계속 진행
                        # (per_agent_results에 반영)
                        for a in self.possible_agents:
                            if a not in [ar for ar in self.possible_agents if ar in self.eliminated_agents and ar != agent]:
                                pass  # 아래 보상 루프에서 합산
                        # 이 에이전트는 제거되었으므로 inner loop break
                        self._pending_mid_reward = mid_reward
                        break

            # 제거된 에이전트는 나머지 체크 건너뛰기
            if agent in self.eliminated_agents:
                continue

            # 이탈 체크
            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location["Drone1"].position.x_val,
                self.current_location[agent].position.y_val - self.current_location["Drone1"].position.y_val,
                self.current_location[agent].position.z_val - self.current_location["Drone1"].position.z_val
            ])

            if _distance_leader > self.far_cutoff:
                print(f"[이탈] {agent} 리더 거리 초과! → 전체 실패")
                return self._end_episode(self.REWARD_FAIL, "FAIL_AGENT_FAR_CUTOFF")

            # 리더와 충돌
            hit, d = self._check_distance_collision(agent, "Drone1", threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"⚠️💔[충돌] {agent} ↔ Leader")
                return self._end_episode(self.REWARD_FAIL, "FAIL_DISTANCE_AGENT_LEADER")

            # 아군끼리 충돌 (활성 아군끼리만)
            for other in active_agents:
                if other == agent or other in self.eliminated_agents:
                    continue
                hit, d = self._check_distance_collision(agent, other, threshold=self.COLLISION_THRESHOLD)
                if hit:
                    print(f"💥🤖[충돌] {agent} ↔ {other}")
                    return self._end_episode(self.REWARD_FAIL, "FAIL_DISTANCE_AGENT_AGENT")

        # 리더 피격 체크 (활성 적기 각각에 대해)
        for dn in self._get_active_enemies():
            collisionInfo = self.client.simGetCollisionInfo("Drone1")
            if collisionInfo.has_collided and collisionInfo.object_name == dn:
                print(f"💥[피격] 리더가 {dn}에게 피격됨!")
                return self._end_episode(self.REWARD_FAIL, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")

        # 활성 아군이 전멸했는데 적이 아직 남아있는 경우
        if len(self._get_active_agents()) == 0 and len(self._get_active_enemies()) > 0:
            print(f"💀[전멸] 모든 아군이 제거됐지만 적기가 남아있음!")
            return self._end_episode(self.REWARD_FAIL, "FAIL_ALL_AGENTS_ELIMINATED")

        # ---- 관측 & 보상 수집 ----
        for agent in self.possible_agents:
            obs = self._get_obs(agent)
            per_agent_obs.append(obs)

            if agent in self.eliminated_agents:
                if self._pending_mid_reward is not None:
                    per_agent_results.append(self._pending_mid_reward.get(agent, 0.0))
                else:
                    per_agent_results.append(0.0)

                per_agent_infos.append({
                    "step_reward": 0.0,
                    "role": "eliminated",
                    "obstacle_states": {dn: self._obstacle_states[dn] for dn in self.dynamic_names}
                })
            else:
                _reward = self._compute_reward(agent)
                if self._pending_mid_reward is not None:
                    _reward += self._pending_mid_reward.get(agent, 0.0)
                per_agent_results.append(_reward)
                per_agent_infos.append({
                    "step_reward": _reward,
                    "role": self.role_map.get(agent, "escort"),
                    "interceptor_map": dict(self.interceptor_map),
                    "obstacle_states": {dn: self._obstacle_states[dn] for dn in self.dynamic_names}
                })

                # lidar 이전 위치 저장
                for dn in self._get_active_enemies():
                    if dn in self.lidar_memory.get(agent, {}):
                        norm_curr = self.lidar_memory[agent][dn]
                        real_curr = [norm_curr[0] * 100.0, norm_curr[1] * 100.0]
                        self.prev_lidar_pos[agent][dn] = real_curr

        termination_list = [False for _ in self.possible_agents]
        rewards_list = self._get_rewards(per_agent_results)
        obs_list = per_agent_obs
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list
