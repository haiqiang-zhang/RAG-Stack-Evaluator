import time


def measure_speed(func, *args, **kwargs):
	"""
	Method for measuring execution speed of the function.
	"""
	start_time = time.time()
	result = func(*args, **kwargs)
	end_time = time.time()
	return result, end_time - start_time
