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

def extract(log_file):
	pattern = re.compile(r'(loss_\w+?(_\d+)?): ([\d\.]+)')
	losses = {}
	with open(log_file, 'r', encoding='utf-8') as f:
		for line in f:
			if ',' not in line:
				continue
			items = line.split(',')
			for item in items:
				if ':' not in item:
					continue
				loss_type, value = item.split(':')
				loss_type = loss_type.strip()
				value = float(value.strip())
				if loss_type not in losses:
					losses[loss_type] = []
				losses[loss_type].append(value)
			
	# 打印每种损失的名称和对应的项数
	# for loss_type, values in losses.items():
	# 	print(f"{loss_type}: {len(values)}")
	return losses

def plot_losses(losses, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	grouped_losses = {}
	for loss_type in losses:
		base_loss_type = re.sub(r'_\d+$', '', loss_type)
		if base_loss_type not in grouped_losses:
			grouped_losses[base_loss_type] = {}
		grouped_losses[base_loss_type][loss_type] = losses[loss_type]

	for base_loss_type, loss_group in grouped_losses.items():
		plt.figure()
		for loss_type, values in loss_group.items():
			plt.plot(values, label=loss_type)
		plt.xlabel('iterations')
		plt.ylabel('loss')
		plt.title(f'Training loss: {base_loss_type}')
		plt.legend()
		output_file = os.path.join(output_dir, f'{base_loss_type}_loss_plot.png')
		plt.savefig(output_file)
		plt.close()

if __name__ == "__main__":
	log_dir = "/data3/ipad_3d/MDP/outputs/monodetr_20240715_161833"
	timestamp = log_dir.split('_')[-2] + '_' + log_dir.split('_')[-1]
	log_file = os.path.join(log_dir, f"train.log.{timestamp}")
	out_dir = os.path.join(log_dir, 'AP_Plots')  # 输出目录
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	bbox_ap_values, bev_ap_values, d3_ap_values, aos_ap_values = extract_car_ap_r40(log_file)
	bbox_ap_values = bbox_ap_values[:-1]
	bev_ap_values = bev_ap_values[:-1]
	d3_ap_values = d3_ap_values[:-1]
	aos_ap_values = aos_ap_values[:-1]
	plot_metric(bbox_ap_values, 'bbox AP', os.path.join(out_dir, 'bbox_ap_curve.png'))
	plot_metric(bev_ap_values, 'bev AP', os.path.join(out_dir, 'bev_ap_curve.png'))
	plot_metric(d3_ap_values, '3d AP', os.path.join(out_dir, '3d_ap_curve.png'))
	plot_metric(aos_ap_values, 'aos AP', os.path.join(out_dir, 'aos_ap_curve.png'))
	loss_log_file = os.path.join(log_dir, "monodetr.log")
	output_dir = os.path.join(log_dir, 'loss_plots')
	losses = extract(loss_log_file)
	plot_losses(losses, output_dir)