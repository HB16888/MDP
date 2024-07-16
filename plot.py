import re
import os
import matplotlib.pyplot as plt

def extract_car_ap_r40(log_file):
	bbox_ap_values = []
	bev_ap_values = []
	d3_ap_values = []
	aos_ap_values = []
	
	with open(log_file, 'r') as file:
		for line in file:
			if 'Car AP_R40@0.70, 0.70, 0.70:' in line:
				# next(file)  # Skip the line with 'bbox AP'
				bbox_line = next(file).strip()
				bev_line = next(file).strip()
				d3_line = next(file).strip()
				aos_line = next(file).strip()
				
				bbox_ap = list(map(float, re.findall(r'\d+\.\d+', bbox_line)))
				bev_ap = list(map(float, re.findall(r'\d+\.\d+', bev_line)))
				d3_ap = list(map(float, re.findall(r'\d+\.\d+', d3_line)))
				aos_ap = list(map(float, re.findall(r'\d+\.\d+', aos_line)))
				
				bbox_ap_values.append(bbox_ap)
				bev_ap_values.append(bev_ap)
				d3_ap_values.append(d3_ap)
				aos_ap_values.append(aos_ap)
	
	return bbox_ap_values, bev_ap_values, d3_ap_values, aos_ap_values

def plot_metric(values, metric_name, output_file):
	epochs = list(range(1, len(values) + 1))
	plt.figure(figsize=(10, 5))
	for i, threshold in enumerate(['Easy', 'Moderate', 'Hard']):
		metric_values = [epoch_values[i] for epoch_values in values]
		plt.plot(epochs, metric_values, marker='o', linestyle='-', label=f'{threshold}')
	plt.title(f'{metric_name} over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel(metric_name)
	plt.legend()
	plt.grid(True)
	plt.savefig(output_file)

if __name__ == "__main__":
	log_file = "/data3/ipad_3d/MDP/outputs/monodetr_20240715_161833/train.log.20240715_161833"  # 日志文件名
	log_dir = os.path.dirname(log_file)  # 获取日志文件所在目录
	
	bbox_ap_values, bev_ap_values, d3_ap_values, aos_ap_values = extract_car_ap_r40(log_file)
	bbox_ap_values = bbox_ap_values[:-1]
	bev_ap_values = bev_ap_values[:-1]
	d3_ap_values = d3_ap_values[:-1]
	aos_ap_values = aos_ap_values[:-1]
	plot_metric(bbox_ap_values, 'bbox AP', os.path.join(log_dir, 'bbox_ap_curve.png'))
	plot_metric(bev_ap_values, 'bev AP', os.path.join(log_dir, 'bev_ap_curve.png'))
	plot_metric(d3_ap_values, '3d AP', os.path.join(log_dir, '3d_ap_curve.png'))
	plot_metric(aos_ap_values, 'aos AP', os.path.join(log_dir, 'aos_ap_curve.png'))