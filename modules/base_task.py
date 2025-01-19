import pygame
from modules.utils import config
sampling_freq = config['simulation']['sampling_freq']
sampling_time = 1.0 / sampling_freq  # in seconds
class BaseTask:
    def __init__(self, id, position, id_type="task"):
        if id_type not in ["task", "sam"]:
            raise ValueError(f"Invalid id_type. Must be one of ['task', 'sam'], but got {id_type}")
        self.position = pygame.Vector2(position)
        self.completed = False
        self.amount = 0.0
        
        if id_type == "task":
            self.task_id = id
            self.sam_id = None
        elif id_type == "sam":
            self.sam_id = id
            self.task_id = None

    def set_done(self):
        self.completed = True

    def reduce_amount(self, work_rate):
        self.amount -= work_rate * sampling_time
        if self.amount <= 0:
            self.set_done()

    def draw(self, screen):
        if not self.completed:
            pygame.draw.circle(screen, self.color, self.position, int(5))

    def draw_task_id(self, screen):
        if not self.completed:
            font = pygame.font.Font(None, 15)
            if self.task_id is not None:
                text_surface = font.render(f"task_id {self.task_id}", True, (50, 50, 50))
            elif self.sam_id is not None:
                text_surface = font.render(f"SAM_id {self.sam_id}", True, (50, 50, 50))
            screen.blit(text_surface, (self.position[0], self.position[1]))

