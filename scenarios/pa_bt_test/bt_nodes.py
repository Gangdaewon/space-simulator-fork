import math
import random
from modules.base_bt_nodes import BTNodeList, Status, Node, Sequence, Fallback, SyncAction, SyncCondition, LocalSensingNode, DecisionMakingNode
from scenarios.pa_bt_test.task import Task, SAM

# BT Node List
CUSTOM_ACTION_NODES = [
    'CompleteTask',
    'MoveToTask',
    'AssignTask',
    'ExploreTask'
    ]

CUSTOM_CONDITION_NODES = [
    'IsMissionCompleted',
    'IsTaskNearby',
    'IsTaskAssigned',
    'IsTaskSensored'
]

BTNodeList.ACTION_NODES.extend(CUSTOM_ACTION_NODES)
BTNodeList.CONDITION_NODES.extend(CUSTOM_CONDITION_NODES)

# Scenario-specific Action/Condition Nodes
from modules.utils import config
target_arrive_threshold = config['tasks']['threshold_done_by_arrival']
task_locations = config['tasks']['locations']
sampling_freq = config['simulation']['sampling_freq']
sampling_time = 1.0 / sampling_freq  # in seconds
agent_max_random_movement_duration = config.get('agents', {}).get('random_exploration_duration', None)


# ===================================
# ============Action Nodes===========
# ===================================

# CompleteTask Node
class CompleteTask(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._complete_task)
        self.agent = agent

    def _complete_task(self, agent, blackboard):
        assigned_task = blackboard.get('assigned_task', None)
        missiles_remained = blackboard.get('missiles_remained', 0)
        if assigned_task is None:
            return Status.FAILURE

        task_position = assigned_task.position
        agent_position = agent.position

        # Check whether work location has been reached
        distance = math.sqrt((task_position[0] - agent_position[0])**2 +
                             (task_position[1] - agent_position[1])**2)
        
        if distance < 5.0:  # Task or SAM is within range
            assigned_task.reduce_amount(agent.work_rate)
            agent.update_task_amount_done(agent.work_rate)

            if not assigned_task.completed:
                return Status.RUNNING

            if isinstance(assigned_task, SAM):
                sam_id = assigned_task.sam_id
                blackboard['missiles_remained'] -= 1
                print(f"Agent {agent.agent_id}: Neutralized SAM {sam_id}. Remaining missiles: {blackboard['missiles_remained']}")
            else:
                task_id = assigned_task.task_id
                blackboard['missiles_remained'] -= 1
                print(f"Agent {agent.agent_id}: Completed task {task_id}. Remaining missiles: {blackboard['missiles_remained']}")

            blackboard['assigned_task'] = None

        remaining_tasks = [task for task in agent.tasks_info if not task.completed]
        if len(remaining_tasks) == 0:
            print("All tasks completed. Mission accomplished!")
            return Status.SUCCESS
        else:
            return Status.FAILURE



# MoveToTask Node
class MoveToTask(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._move_to_task)
        self.agent = agent

    def _move_to_task(self, agent, blackboard):
        assigned_task = blackboard.get('assigned_task', None)
        if assigned_task is None:
            return Status.FAILURE

        task_position = assigned_task.position
        agent_position = agent.position
        agent.follow(task_position)
        distance = math.sqrt((task_position[0] - agent_position[0])**2 +
                             (task_position[1] - agent_position[1])**2)

        if distance < 5.0:
            print(f"Arrived at Task ID: {assigned_task.task_id}")
            return Status.SUCCESS

        return Status.RUNNING



# AssignTask Node
class AssignTask(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._assign_task)
        self.agent = agent

    def _assign_task(self, agent, blackboard):
        nearby_sams = blackboard.get('nearby_sams', [])  # 탐지된 SAM 리스트
        missiles_remained = blackboard.get('missiles_remained', 0)
        assigned_task = blackboard.get('assigned_task', None)

        # Step 1: Task와 SAM이 모두 없으면 FAILURE 반환
        tasks = blackboard.get('local_tasks_info', [])
        if len(tasks) == 0 and len(nearby_sams) == 0:
            print(f"Agent {agent.agent_id}: No tasks or SAMs available for assignment.")
            return Status.FAILURE

        # Step 2: SAM 할당 로직
        if missiles_remained > 1 and nearby_sams:
            # 가장 가까운 SAM 찾기
            closest_sam = min(
                (sam for sam in nearby_sams if not sam.completed),  # SAM 객체 리스트
                key=lambda sam: (agent.position - sam.position).length_squared(),
                default=None
            )
            if closest_sam:
                # SAM을 우선적으로 할당
                if not assigned_task or isinstance(assigned_task, Task):  # 기존 Task보다 SAM 우선
                    blackboard['assigned_task'] = closest_sam
                    print(f"Agent {agent.agent_id}: Assigned SAM {closest_sam.sam_id} as target. Remaining missiles: {missiles_remained}")
                    return Status.SUCCESS

        # Step 3: 가장 가까운 Task 찾기
        agent_position = agent.position
        closest_task = None
        closest_distance = float('inf')

        for task in tasks:
            distance = math.sqrt(
                (task.position[0] - agent_position[0])**2 +
                (task.position[1] - agent_position[1])**2
            )
            if distance < closest_distance:
                closest_distance = distance
                closest_task = task

        # Task 할당
        if closest_task:
            blackboard['assigned_task'] = closest_task
            print(f"Agent {agent.agent_id}: Assigned Task {closest_task.task_id}.")
            return Status.SUCCESS

        # 모든 조건이 실패하면 FAILURE 반환
        return Status.FAILURE




# Exploration node
class ExploreTask(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._random_explore)
        self.random_move_time = float('inf')
        self.random_waypoint = (0, 0)

    def _random_explore(self, agent, blackboard):
        # Move towards a random position
        if self.random_move_time > agent_max_random_movement_duration:
            self.random_waypoint = self.get_random_position(task_locations['x_min'], task_locations['x_max'], task_locations['y_min'], task_locations['y_max'])
            self.random_move_time = 0 # Initialisation
        
        blackboard['random_waypoint'] = self.random_waypoint        
        self.random_move_time += sampling_time   
        agent.follow(self.random_waypoint)         
        return Status.RUNNING
        
    def get_random_position(self, x_min, x_max, y_min, y_max):
        pos = (random.randint(x_min, x_max),
                random.randint(y_min, y_max))
        return pos
    


# ===================================
# ==========Condition Nodes==========
# ===================================

# IsMissionCompleted Node
class IsMissionCompleted(SyncCondition):
    def __init__(self, name, agent):
        super().__init__(name, self._check_mission_completed)

    def _check_mission_completed(self, agent, blackboard):
        # 남은 작업 확인
        remaining_tasks = [task for task in agent.tasks_info if not task.completed]

        if len(remaining_tasks) == 0:
            print("All tasks are completed. Mission accomplished!")
            return Status.SUCCESS  # All tasks are completed
        else:
            return Status.FAILURE



class IsTaskNearby(SyncCondition):
    def __init__(self, name, agent):
        super().__init__(name, self._is_task_nearby)
    
    def _is_task_nearby(self, agent, blackboard):
        assigned_task = blackboard.get('assigned_task', None)
        if assigned_task is None:
            return Status.FAILURE  # Return FAILURE if no tasks are available

        task_position = assigned_task.position
        agent_position = agent.position
        distance = math.sqrt((task_position[0] - agent_position[0])**2 +
                             (task_position[1] - agent_position[1])**2)

        if distance < 5.0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class IsTaskAssigned(SyncCondition):
    def __init__(self, name, agent):
        super().__init__(name, self._is_task_assigned)

    def _is_task_assigned(self, agent, blackboard):
        assigned_task = blackboard.get('assigned_task', None)
        if assigned_task is not None:
            return Status.SUCCESS
        return Status.FAILURE




class IsTaskSensored(SyncCondition):
    def __init__(self, name, agent):
        super().__init__(name, self._is_task_sensored)

    def _is_task_sensored(self, agent, blackboard):
        local_tasks_info = blackboard.get('local_tasks_info', [])
        nearby_sams = blackboard.get('nearby_sams', [])
        return Status.SUCCESS if len(local_tasks_info) > 0 or len(nearby_sams) > 0 else Status.FAILURE