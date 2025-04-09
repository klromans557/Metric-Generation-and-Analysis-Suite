import os
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
import tkinter.filedialog as filedialog 

class ScriptRunnerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("reaper557's Metric Generation and Analysis Suite (M.G.A.S.) - v3.0")
        self.geometry("750x1000")
        self.configure(bg='#7a7a7a')  # Change the background color of the main window
        
        self.default_images_dir = os.path.join(os.getcwd(), 'DIR', 'images')
        self.default_references_dir = os.path.join(os.getcwd(), 'DIR', 'references')
        self.default_output_dir = os.path.join(os.getcwd(), 'DIR', 'output')
        self.default_logs_dir = os.path.join(os.getcwd(), 'LOGS')
        self.default_dlib_dir = os.path.join(os.getcwd(), 'DLIB')

        self.log_update_interval = 1000
        self.is_running_first_script = False

        self.plot_queue = queue.Queue()

        self.create_widgets()
        self.check_plot_queue()

    def create_widgets(self):       
        # Add a logo image at the top
        logo_path = os.path.join(os.getcwd(), '_EXTRAS', 'Grim_AI-Enhanced.png')
        self.logo_image = Image.open(logo_path)
        self.logo_image = self.logo_image.resize((100, 100), Image.Resampling.LANCZOS)  # Resize the image
        self.logo = ImageTk.PhotoImage(self.logo_image)
        self.logo_label = tk.Label(self, image=self.logo, bg='#f0f0f0')
        self.logo_label.pack(pady=10)
        
        # Main sections with background color
        section_bg_color = '#bfbfbf'
        self.section1 = tk.Frame(self, bg=section_bg_color)
        self.section1.pack(pady=10, padx=10, fill=tk.X)

        self.section2 = tk.Frame(self, bg=section_bg_color)
        self.section2.pack(pady=10, padx=10, fill=tk.X)

        self.section3 = tk.Frame(self, bg=section_bg_color)
        self.section3.pack(pady=10, padx=10, fill=tk.X)

        self.section4 = tk.Frame(self, bg=section_bg_color)
        self.section4.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Section 1: Directory options (2x4 grid)
        self.images_dir_label = tk.Label(self.section1, text="Images Directory")
        self.images_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.images_dir = tk.Entry(self.section1)
        self.images_dir.insert(0, self.default_images_dir)
        self.images_dir.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.images_dir_button = tk.Button(self.section1, text="Change", command=self.change_images_dir)
        self.images_dir_button.grid(row=0, column=2, padx=5, pady=5)

        self.references_dir_label = tk.Label(self.section1, text="References Directory")
        self.references_dir_label.grid(row=0, column=3, padx=5, pady=5, sticky="e")
        self.references_dir = tk.Entry(self.section1)
        self.references_dir.insert(0, self.default_references_dir)
        self.references_dir.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.references_dir_button = tk.Button(self.section1, text="Change", command=self.change_references_dir)
        self.references_dir_button.grid(row=0, column=5, padx=5, pady=5)

        self.output_dir_label = tk.Label(self.section1, text="Output Directory")
        self.output_dir_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.output_dir = tk.Entry(self.section1)
        self.output_dir.insert(0, self.default_output_dir)
        self.output_dir.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.output_dir_button = tk.Button(self.section1, text="Change", command=self.change_output_dir)
        self.output_dir_button.grid(row=1, column=2, padx=5, pady=5)

        self.logs_dir_label = tk.Label(self.section1, text="Logs Directory")
        self.logs_dir_label.grid(row=1, column=3, padx=5, pady=5, sticky="e")
        self.logs_dir = tk.Entry(self.section1)
        self.logs_dir.insert(0, self.default_logs_dir)
        self.logs_dir.grid(row=1, column=4, padx=5, pady=5, sticky="w")
        self.logs_dir_button = tk.Button(self.section1, text="Change", command=self.change_logs_dir)
        self.logs_dir_button.grid(row=1, column=5, padx=5, pady=5)

        # Section 2: Script variables (1x4 grid)
        self.num_processes_label = tk.Label(self.section2, text="Number of Processes")
        self.num_processes_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.num_processes = tk.Entry(self.section2)
        self.num_processes.insert(0, "2")
        self.num_processes.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.models_to_compare_label = tk.Label(self.section2, text="Models to Compare (comma-separated)")
        self.models_to_compare_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.models_to_compare = tk.Entry(self.section2)
        self.models_to_compare.insert(0, "1,2")
        self.models_to_compare.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Section 3: Buttons (1x4 grid)
        self.run_face_button = tk.Button(self.section3, text="Run Face Script", command=self.run_create_facedistance_data)
        self.run_face_button.grid(row=0, column=0, padx=5, pady=5)

        self.run_style_button = tk.Button(self.section3, text="Run Style Script", command=self.run_create_styledata)
        self.run_style_button.grid(row=0, column=1, padx=5, pady=5)

        self.run_bulk_button = tk.Button(self.section3, text="Run Bulk Script", command=self.run_bulk_facedistance_statistics)
        self.run_bulk_button.grid(row=0, column=2, padx=5, pady=5)

        self.open_last_figure_button = tk.Button(self.section3, text="Open Last Figure", command=self.open_saved_figure)
        self.open_last_figure_button.grid(row=0, column=3, padx=5, pady=5)

        # Section 4: Output text box
        self.output_text = tk.Text(self.section4, state="disabled")
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def change_images_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.default_images_dir, title="Select Images Directory")
        if new_dir:
            if self.is_directory_empty(new_dir):
                messagebox.showerror("Error", "The selected images directory is empty.")
                return
            self.images_dir.delete(0, tk.END)
            self.images_dir.insert(0, new_dir)

    def change_references_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.default_references_dir, title="Select References Directory")
        if new_dir:
            if self.is_directory_empty(new_dir):
                messagebox.showerror("Error", "The selected references directory is empty.")
                return
            self.references_dir.delete(0, tk.END)
            self.references_dir.insert(0, new_dir)

    def change_output_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.default_output_dir, title="Select Output Directory")
        if new_dir:
            if self.is_directory_empty(new_dir):
                messagebox.showerror("Error", "The selected output directory is empty.")
                return
            self.output_dir.delete(0, tk.END)
            self.output_dir.insert(0, new_dir)

    def change_logs_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.default_logs_dir, title="Select Logs Directory")
        if new_dir:
            self.logs_dir.delete(0, tk.END)
            self.logs_dir.insert(0, new_dir)

    def run_script(self, script_name, output_file=None):
        # Set the log update interval based on the script name
        if script_name == 'create_facedistance_data.py':
            self.log_update_interval = 1000
        elif script_name == 'create_styledata.py':
            self.log_update_interval = 1000
        elif script_name == 'bulk_facedistance_statistics_v2.py':
            self.log_update_interval = 10000
        
        # Clear the output text box
        self.output_text.configure(state="normal")
        self.output_text.delete('1.0', tk.END)
        self.output_text.configure(state="disabled")

        # Determine the current log file based on the script being run
        if script_name == 'create_facedistance_data.py':
            self.current_log_file = 'process_log.txt'
            log_file_path = os.path.join(self.logs_dir.get(), self.current_log_file)
            with open(log_file_path, 'w') as log_file:
                log_file.write('Starting new run...\n')
        elif script_name == 'bulk_facedistance_statistics_v2.py':
            self.current_log_file = 'output_stats.txt'
            output_stats_path = os.path.join(self.logs_dir.get(), self.current_log_file)
            with open(output_stats_path, 'w') as output_stats_file:
                output_stats_file.write('')
        elif script_name == 'create_styledata.py':
            self.current_log_file = 'styledata_log.txt'
            style_log_path = os.path.join(self.logs_dir.get(), self.current_log_file)
            with open(style_log_path, 'w') as style_log_file:
                style_log_file.write('Starting style script log.\n')

        # Fix local venv assignment
        #command = ['python', script_name]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(script_dir, 'venv', 'Scripts', 'python.exe')
        command = [venv_python, script_name]

        env = os.environ.copy()
        env['IMAGES_DIR'] = self.images_dir.get()
        env['REFERENCES_DIR'] = self.references_dir.get()
        env['OUTPUT_DIR'] = self.output_dir.get()
        env['NUM_PROCESSES'] = str(self.num_processes.get())
        env['MODELS_TO_COMPARE'] = self.models_to_compare.get()

        def run():
            try:
                plt.ioff()  # Disable interactive mode
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()  # Collect output and errors
                rc = process.returncode
                if rc != 0:
                    self.display_output(stderr)
                    messagebox.showerror("Error", f"Error running {script_name}: {stderr}")
                else:
                    if output_file:
                        with open(output_file, 'r') as file:
                            output_content = file.read()
                        self.display_output(output_content)
                    else:
                        self.display_output(stdout)
                    self.plot_queue.put('draw')  # Signal to draw the plot
            except Exception as e:
                self.display_output(str(e))
                messagebox.showerror("Error", f"Error running {script_name}: {e}")

        threading.Thread(target=run).start()
        self.update_output()

    def update_output(self):
        try:
            log_file_path = os.path.join(self.logs_dir.get(), self.current_log_file)
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as log_file:
                    log_content = log_file.read()
                    self.output_text.configure(state="normal")
                    self.output_text.delete('1.0', tk.END)
                    self.output_text.insert(tk.END, log_content)
                    self.output_text.configure(state="disabled")
                    self.output_text.see(tk.END)
        except Exception as e:
            print(f"Error reading log file: {e}")
        finally:
            self.after(self.log_update_interval, self.update_output)

    def check_plot_queue(self):
        try:
            while not self.plot_queue.empty():
                item = self.plot_queue.get()
                if item == 'draw':
                    self.draw_plot()
        except Exception as e:
            print(f"Error processing plot queue: {e}")
        finally:
            self.after(100, self.check_plot_queue)

    def draw_plot(self):
        try:
            # The figure has already been saved in the bulk script
            # No need to create or show the plot again here
            pass
        except Exception as e:
            messagebox.showerror("Error", f"Error drawing plot: {e}")

    def open_saved_figure(self):
        figure_path = os.path.join(os.getcwd(), '_EXTRAS', 'comparison_plot.png')
        if os.path.exists(figure_path):
            try:
                if os.name == 'posix':
                    subprocess.call(['xdg-open', figure_path])
                else:
                    os.startfile(figure_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open figure file: {e}")
        else:
            messagebox.showerror("Error", "Figure file does not exist.")

    def is_directory_empty(self, directory):
        return not any(os.scandir(directory))

    def run_create_facedistance_data(self):
        if self.is_directory_empty(self.images_dir.get()):
            messagebox.showerror("Error", "The 'images' folder is empty.")
            return
        if self.is_directory_empty(self.references_dir.get()):
            messagebox.showerror("Error", "The 'references' folder is empty.")
            return
        self.current_log_file = "process_log.txt"
        self.run_script('create_facedistance_data.py')
    
    def run_create_styledata(self):
        if self.is_directory_empty(self.images_dir.get()):
            messagebox.showerror("Error", "The 'images' folder is empty.")
            return
        if self.is_directory_empty(self.references_dir.get()):
            messagebox.showerror("Error", "The 'references' folder is empty.")
            return
        self.current_log_file = "styledata_log.txt"
        self.run_script('create_styledata.py')

    def run_bulk_facedistance_statistics(self):
        if self.is_directory_empty(self.output_dir.get()):
            messagebox.showerror("Error", "The 'output' folder is empty.")
            return
        output_stats_file = os.path.join(self.logs_dir.get(), 'output_stats.txt')
        self.current_log_file = "output_stats.txt"
        self.run_script('bulk_facedistance_statistics_v2.py', output_file=output_stats_file)

    def display_output(self, output):
        self.output_text.configure(state="normal")
        self.output_text.insert("end", output)
        self.output_text.configure(state="disabled")
        self.output_text.see("end")

if __name__ == "__main__":
    app = ScriptRunnerApp()
    app.mainloop()
