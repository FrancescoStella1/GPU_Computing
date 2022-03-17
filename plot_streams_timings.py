from re import A
import numpy as np
import matplotlib.pyplot as plt


FONTSIZE = 15
LABELS_SIZE = 12
TITLE_SIZE = 16

SAVE_FIG = False


def plot_stats(dct1, dct2, dct3, x_label, y_label, title):
	r"""
	Plots average execution times from the three input dicts.

	Parameters
	----------
	dct1: dict
		First dict with key/values as those returned from the function 'read_from_file()'.
	dct2: dict
		Second dict with key/values as those returned from the function 'read_from_file()'.
	dct3: dict
		Third dict with key/values as those returned from the function 'read_from_file()'.
	x_label: str
		X axis label.
	y_label: str
		Y axis label.
	title: str
		Title of the plot.
	"""
	fig, ax = plt.subplots(figsize=(19.20, 10.80))
	labels = ['Grayscale', 'Gamma', 'Gradients (only null-stream)', 'Magnitude/Direction', 'HOG (only null-stream)']
	x = np.arange(5)
	width = 0.2  		# width of the bars
	
	r1 = ax.bar(x - 3*(width/2), list(dct1.values())[1:], width, label=f"{dct1['num_streams']} stream (null-stream)")
	r2 = ax.bar(x - width/2, list(dct2.values())[1:], width, label=f"{dct2['num_streams']} streams")
	r3 = ax.bar(x + width/2, list(dct3.values())[1:], width, label=f"{dct3['num_streams']} streams")

	ax.set_xticks(x - (width/2), labels, fontsize=FONTSIZE)
	ax.legend(fontsize=LABELS_SIZE)

	ax.set_xlabel(x_label, fontsize=FONTSIZE)
	ax.set_ylabel(y_label, fontsize=FONTSIZE)
	ax.set_title(title, fontsize=TITLE_SIZE)

	ax.bar_label(r1, padding=12, fontsize=LABELS_SIZE)
	ax.bar_label(r2, padding=12, fontsize=LABELS_SIZE)
	ax.bar_label(r3, padding=12, fontsize=LABELS_SIZE)
	fig.tight_layout()

	if SAVE_FIG:
		plt.savefig('streams_timings.png', format='png', dpi=600)
	plt.show()


def read_from_file(filename, n_streams):
	r"""
	Reads stats from timing file.

	Parameters
	----------
	filename: str
		Name of the file to read.
	n_streams: int
		Number of streams used to execute the algorithm.

	Returns
	-------
	dct: dict
		Dict containing as keys/values \
			|- {num_streams: n} : n is the number of streams.\
			|- {grayscale: t} : t is the average execution time for grayscale conversion.\
			|- {gamma: t} : t is the average execution time for gamma correction.\
			|- {gradients: t} : t is the average execution time for gradients computation.\
			|- {magdir: t} : t is the average execution time for magnitude and direction computation.\
			|- {hog: t} : t is the average execution time for HOG computation.
	"""
	with open(filename, 'r') as f:
		data = f.read()
	dct = {'num_streams': n_streams}
	timings = data.split("###")
	num = len(timings)
	grayscale_avg = 0
	gamma_avg = 0
	gradients_avg = 0
	magdir_avg = 0
	hog_avg = 0
	for timing in timings:
		stats = timing.split('\n')
		stats = list(filter(lambda x: x!='', stats))
		if len(stats) == 0:
			continue
		grayscale_avg += float(stats[1])
		gamma_avg += float(stats[3])
		gradients_avg += float(stats[5])
		magdir_avg += float(stats[7])
		hog_avg += float(stats[9])
	
	trunc = 5
	dct['grayscale'] = round(grayscale_avg/num, trunc)
	dct['gamma'] = round(gamma_avg/num, trunc)
	dct['gradients'] = round(gradients_avg/num, trunc)
	dct['magdir'] = round(magdir_avg/num, trunc)
	dct['hog'] = round(hog_avg/num, trunc)
	return dct



if __name__=='__main__':
	one_stream = read_from_file('timing_gpu_1.txt', 1)
	two_streams = read_from_file('timing_gpu_2.txt', 2)
	four_streams = read_from_file('timing_gpu_4.txt', 4)
	plot_stats(one_stream, two_streams, four_streams, 'Steps of the algorithm', 'Execution time (s)', 'Average execution time with different number of streams')