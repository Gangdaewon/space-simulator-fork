import pygame
import random
from modules.utils import config, generate_positions, generate_task_colors
dynamic_task_generation = config['tasks'].get('dynamic_task_generation', {})
max_generations = dynamic_task_generation.get('max_generations', 0) if dynamic_task_generation.get('enabled', False) else 0
tasks_per_generation = dynamic_task_generation.get('tasks_per_generation', 0) if dynamic_task_generation.get('enabled', False) else 0

task_colors = generate_task_colors(config['tasks']['quantity'] + tasks_per_generation*max_generations)

from modules.base_task import BaseTask


class Task(BaseTask):
    def __init__(self, task_id, position):
        super().__init__(id = task_id, position = position, id_type="task")
        self.amount = random.uniform(config['tasks']['amounts']['min'], config['tasks']['amounts']['max'])        
        self.radius = self.amount / config['simulation']['task_visualisation_factor']
        self.color = task_colors.get(self.task_id, (0, 0, 0))  # Default to black if task_id not found


    def draw(self, screen):
        self.radius = self.amount / config['simulation']['task_visualisation_factor']        
        if not self.completed:
            pygame.draw.circle(screen, self.color, self.position, int(self.radius))

    def draw_task_id(self, screen):
        if not self.completed:
            font = pygame.font.Font(None, 15)
            text_surface = font.render(f"task_id {self.task_id}: {self.amount:.2f}", True, (250, 250, 250))
            screen.blit(text_surface, (self.position[0], self.position[1]))
            
class SAM(BaseTask):
    def __init__(self, sam_id, position):
        super().__init__(id = sam_id, position = position, id_type="sam")
        self.active = True
        self.amount = random.uniform(config['sams']['amounts']['min'], config['sams']['amounts']['max'])        
        self.radius = self.amount / config['simulation']['sam_visualisation_factor']
        self.detection_range = config['sams']['detection_range']
        self.color = task_colors.get(self.sam_id, (255, 0, 0))
        
    # 시나리오 상 SAM을 타격했을 때 처리에 필요할 경우를 위해 추가
    def deactivate(self):
        self.active = False
        self.color = (0, 0, 0)
        
    def draw(self, screen):
        if self.active:
            pygame.draw.circle(screen, self.color, self.position, int(self.detection_range), 1)
            pygame.draw.circle(screen, self.color, self.position, int(self.radius))
            
    def draw_task_id(self, screen):
        if self.active:
            font = pygame.font.Font(None, 15)
            text_surface = font.render(f"SAM_id {self.sam_id}: {self.amount:.2f}", True, (250, 250, 250))
            screen.blit(text_surface, (self.position[0], self.position[1]))

def generate_tasks(task_quantity=None, task_id_start = 0):
    if task_quantity is None:
        task_quantity = config['tasks']['quantity']        
    task_locations = config['tasks']['locations']

    tasks_positions = generate_positions(task_quantity,
                                        task_locations['x_min'],
                                        task_locations['x_max'],
                                        task_locations['y_min'],
                                        task_locations['y_max'],
                                        radius=task_locations['non_overlap_radius'])

    # Initialize tasks
    tasks = [Task(idx + task_id_start, pos) for idx, pos in enumerate(tasks_positions)]
    return tasks

def generate_sams(sam_quantity=None, sam_id_start = 0):
    if sam_quantity is None:
        sam_quantity = config['sams']['quantity']        
    sam_locations = config['sams']['locations']

    sams_positions = generate_positions(sam_quantity,
                                        sam_locations['x_min'],
                                        sam_locations['x_max'],
                                        sam_locations['y_min'],
                                        sam_locations['y_max'],
                                        radius=sam_locations['non_overlap_radius'])

    # Initialize sams
    sams = [SAM(idx + sam_id_start, pos) for idx, pos in enumerate(sams_positions)]
    return sams
