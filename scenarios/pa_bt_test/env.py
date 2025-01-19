import pygame
from modules.base_env import BaseEnv, pre_render_text
from modules.utils import ResultSaver
from scenarios.pa_bt_test.task import generate_tasks, generate_sams
from scenarios.pa_bt_test.agent import generate_agents

from modules.base_bt_nodes import Status
from modules.ppa_bt_constructor import load_library, expand_behavior_tree


class Env(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

        # Load PPA library from the CSV file
        ppa_library_path = config['simulation'].get('ppa_library_path', 'ppa_library.csv')
        self.ppa_library = load_library(ppa_library_path)

        # Initialize agents and tasks
        self.tasks = generate_tasks()
        self.sams = generate_sams() # 시나리오 상 SAM 추가
        self.agents = generate_agents(self.tasks, self.sams) # sam과 task 정보를 통해 agent 생성(오류 발생x?, paraemter 1개만 전달 받을텐데)
        self.tasks_left = len(self.tasks)  # 초기 작업 수를 설정
        
        # Set `generate_tasks` function for dynamic task generation
        self.generate_tasks = generate_tasks
        self.generate_sams = generate_sams
        
        # Initialize data recording
        self.data_records = []
        self.result_saver = ResultSaver(config)
        
    def generate_sams_if_needed(self):
        """Generate new SAMs dynamically based on configuration."""
        dynamic_sam_generation = self.config['sams'].get('dynamic_task_generation', {})
        sam_per_generation = dynamic_sam_generation.get('tasks_per_generation', 0)
        max_generations = dynamic_sam_generation.get('max_generations', 0)

        if self.generation_count < max_generations:
            if self.simulation_time - self.last_generation_time >= self.generation_interval:
                new_sam_id_start = len(self.sams)
                new_sams = self.generate_sams(sam_quantity=sam_per_generation, sam_id_start=new_sam_id_start)
                self.sams.extend(new_sams)
                self.last_generation_time = self.simulation_time
                self.generation_count += 1
                if self.rendering_mode != "None":
                    print(f"[{self.simulation_time:.2f}] Added {sam_per_generation} new SAMs: Generation {self.generation_count}.")
           
    def save_results(self):
        # Save gif
        if self.save_gif and self.rendering_mode == "Screen":        
            self.recording = False
            print("Recording stopped.")
            self.result_saver.save_gif(self.frames)          
     

        # Save time series data
        if self.save_timewise_result_csv:        
            csv_file_path = self.result_saver.save_to_csv("timewise", self.data_records, ['time', 'agents_total_distance_moved', 'agents_total_task_amount_done', 'remaining_tasks', 'tasks_total_amount_left'])          
            self.result_saver.plot_timewise_result(csv_file_path)
        
        # Save agent-wise data            
        if self.save_agentwise_result_csv:        
            variables_to_save = ['agent_id', 'task_amount_done', 'distance_moved']
            agentwise_results = self.result_saver.get_agentwise_results(self.agents, variables_to_save)                        
            csv_file_path = self.result_saver.save_to_csv('agentwise', agentwise_results, variables_to_save)
            
            self.result_saver.plot_boxplot(csv_file_path, variables_to_save[1:])

        # Save yaml: TODO - To debug
        # if self.save_config_yaml:                
            # self.result_saver.save_config_yaml()           
   
    def record_timewise_result(self):
        agents_total_distance_moved = sum(agent.distance_moved for agent in self.agents)
        agents_total_task_amount_done = sum(agent.task_amount_done for agent in self.agents)
        remaining_tasks = len([task for task in self.tasks if not task.completed])
        tasks_total_amount_left = sum(task.amount for task in self.tasks)
        
        self.data_records.append([
            self.simulation_time, 
            agents_total_distance_moved,
            agents_total_task_amount_done,
            remaining_tasks,
            tasks_total_amount_left
        ])        

    # base_env에서 override           
    async def step(self):
        # Main simulation loop logic
        for agent in self.agents:
            agent.get_sams_nearby()
            result = await agent.run_tree()
            
            if result == Status.FAILURE:  # Check if the result is FAILURE
                failed_conditions = agent.find_failed_conditions()  # Identify failed conditions
                
                for failed_condition in failed_conditions:
                    # Expand the behavior tree based on the failed condition
                    agent.tree = expand_behavior_tree(agent.tree, failed_condition, self.ppa_library)

            agent.update()
        # 필요 시, 디버그 추가, agent.blackboard 활용. (대신 last agent인 경우만)

        # Status retrieval
        self.simulation_time += self.sampling_time
        self.tasks_left = sum(1 for task in self.tasks if not task.completed)
        if self.tasks_left == 0:
            self.mission_completed = not self.generation_enabled or self.generation_count == self.max_generations

        # Dynamic task generation
        if self.generation_enabled:
            self.generate_tasks_if_needed()
            self.generate_sams_if_needed()


        # Stop if maximum simulation time reached
        if self.max_simulation_time > 0 and self.simulation_time > self.max_simulation_time:
            self.running = False
            
    def render(self):
        if self.rendering_mode == "Screen" and self.screen:
            # Draw background
            self.draw_background()

            # Draw agents & their auxiliary information
            self.draw_agents_info()
            self.draw_agents()

            # Draw tasks & their auxiliary information
            self.draw_tasks()
            self.draw_tasks_info()

            # Draw SAMs and their information
            for sam in self.sams:
                sam.draw(self.screen)
                sam.draw_task_id(self.screen)

            # Display task quantity and elapsed simulation time
            task_time_text = pre_render_text(f'Tasks left: {self.tasks_left}; Time: {self.simulation_time:.2f}s', 36, (0, 0, 0))
            self.screen.blit(task_time_text, (self.screen_width - 350, 20))

            # Check if all tasks are completed
            if self.mission_completed:
                text_rect = self.mission_completed_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
                self.screen.blit(self.mission_completed_text, text_rect)

            pygame.display.flip()
            self.clock.tick(self.sampling_freq * self.speed_up_factor)

        elif self.rendering_mode == "Terminal":
            print(f"Time: {self.simulation_time:.2f}, Tasks left: {len(self.tasks)}")
            if self.simulation_time - self.last_print_time > 0.5:
                self.last_print_time = self.simulation_time

            if self.mission_completed:
                print(f'MISSION COMPLETED')
                self.running = False

        else:  # if rendering_mode is None
            if self.mission_completed:
                print(f'[{self.simulation_time:.2f}] MISSION COMPLETED')
                self.running = False