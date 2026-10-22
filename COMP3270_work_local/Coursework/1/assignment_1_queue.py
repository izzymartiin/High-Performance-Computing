import matplotlib.pyplot as plt
import numpy as np
import time
import sys

def simulateQueue(T_final, lambda_arrival, mu_service, C, runs=1):

	sigma = 1 
	time_step = 0.01
	N = int((T_final/time_step) + 1)
	all_queues = np.zeros((runs, N))
	time_points = np.arange(0, T_final + time_step, time_step)

	start = time.perf_counter()

	plt.figure()
	
	for r in range(0, runs):
		q = np.zeros(N)
		for t in range(0, N-1):
			random = np.random.normal(0, 1)
			next_value = (
				(lambda_arrival - mu_service*min(q[t],C))*time_step
				+ sigma*np.sqrt(time_step)*random
				+ q[t]
			)
			if next_value < 0:
				next_value = 0

			q[t+1] = next_value
		all_queues[r] = q

		plt.plot(time_points, q, linewidth=0.8, alpha=0.7)

	end = time.perf_counter()

	# Define the plot elements like axis labels
	plt.xlabel("Time (minutes)")
	plt.ylabel("Queue length")
	plt.title("SDE queue simulation ("+ str(runs) + " runs)")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()


	# Export plot as a PNG image
	plt.savefig("queue_runs.png", dpi=300)
	plt.close()

	#Calculating Queue Statistics
	mean_queue_length = np.mean(np.mean(all_queues, axis=0))
	std_queue_length = np.mean(np.std(all_queues, axis=0))
	mean_queue_occupancy = np.mean(np.mean(all_queues/C, axis=1)) * 100

	print("Mean:", round(mean_queue_length, 2))
	print("Std:", round(std_queue_length, 2))
	print("Mean Occupancy", round(mean_queue_occupancy, 2), "%")
	print("Execution Time:", round((end-start), 2))

	return

simulateQueue(120, 3.33, 0.1, 32, int(sys.argv[1]))
