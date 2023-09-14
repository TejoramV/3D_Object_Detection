import subprocess

num_data = 1000
bounding_box_visualization = False
for iteration in range(num_data):
    command = ["python", "data_gen.py", "--iteration", str(iteration), "--bounding_box_visualization", str(bounding_box_visualization)]
    subprocess.call(command)
