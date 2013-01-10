import string
import sys
import os
import random

amount_of_numbers = 500
data = []
data_len = 0
use_numbers = False
	
def add_to_result_list(result, data):
	result.append(data[0])
	del data[0]
		
def insertion_sort(data):
	for i in xrange(1, len(data)):
		item = data[i]
		item_location = i
		
		while item_location > 0 and data[item_location - 1] > item:
			data[item_location] = data[item_location - 1]
			item_location = item_location - 1
			
		data[item_location] = item
		
def merge(left_data, right_data):
	result = []
	# left_len = len(left_data)
	# right_len = len(right_data)
	
	while len(left_data) > 0 or len(right_data) > 0:
		if len(left_data) > 0 and len(right_data) > 0:
			if left_data[0] <= right_data[0]:
				add_to_result_list(result, left_data)
			else:
				add_to_result_list(result, right_data)
		elif len(left_data) > 0:
			add_to_result_list(result, left_data)
		elif len(right_data) > 0:
			add_to_result_list(result, right_data)
	
	return result
				
def merge_sort(data):
	data_len = len(data)

	if data_len <= 1:
		return data
	
	left = []
	right = []
	middle_location = data_len / 2
	
	for i in xrange(0, middle_location):
		left.append(data[i])
		
	for i in xrange(middle_location, data_len):
		right.append(data[i])
	# print "Left before sort: "
	# print_data(left)
	# raw_input()
	# print "Right before sort: "
	# print_data(right)
	# raw_input()
	left = merge_sort(left)
	right = merge_sort(right)
	# print "Left after sort: "
	# print_data(left)
	# raw_input()
	# print "Right after sort: "
	# print_data(right)
	# raw_input()
	return merge(left, right)

def print_data(data):
	counter = 1
	
	#print "\n"
	
	for i in data:
		sys.stdout.write(str(i) + " ")
		if counter % 10 == 0 and counter != 0:
			print "\n"
		counter += 1
		
	#print "\n"
	print "\n"
		
def heap_sort(data):
	heapify(data)
	#data_len = length(data)
	
	end = data_len - 1
	
	while end > 0:
		swap_elements(end, 0, data)
		end -= 1
		sift_down(0, end, data)

def heapify(data):
	start = (data_len - 2) / 2
	
	while start >= 0:
		sift_down(start, data_len - 1, data)
		start = start - 1
		
def randomize(data):
	del data[:]
	for i in xrange(0, amount_of_numbers):
		if(use_numbers):
			data.append(random.randint(10, 99))
		else:
			mystr = ''.join(random.choice(string.letters))
			data.append(mystr)
			
	data_len = len(data)
			
def selection_sort(data):
	for i in xrange(0, len(data)):
		minimum = data[i]
		minimum_index = i
		original_value = data[i]
		for j in range(i, len(data)):
			if(data[j] < minimum):
				minimum = data[j]
				minimum_index = j
		if(minimum_index != i):
			data[i] = minimum
			data[minimum_index] = original_value

def sift_down(start, end, data):
	root = start
	
	while root * 2 + 1 <= end:
		child = root * 2 + 1
		swap = root
		
		if data[swap] < data[child]:
			swap = child
			
		if child + 1 <= end and data[swap] < data[child + 1]:
			swap = child + 1
			
		if swap != root:
			swap_elements(root, swap, data)
			root = swap
		else:
			return
			
def swap_elements(a, b, data):
	temp = data[a]
	data[a] = data[b]
	data[b] = temp

def test_result(algo):
	if not result:
		print "%s: The following is not sorted!!" % algo
		print_data(data)
		exit(1)
	
def verify_data(data):
	for i in xrange(0, data_len - 1):
		if not data[i] <= data[i + 1]:
			return False
		
	return True
	
for i in xrange(0, 100):
	randomize(data)
	
	heap_sort(data)
	result = verify_data(data)
	test_result("Heap Sort")
	
	randomize(data)
	
	insertion_sort(data)
	result = verify_data(data)
	test_result("Insertion Sort")
	
	randomize(data)
	
	selection_sort(data)
	result = verify_data(data)
	test_result("Selection Sort")
	
	randomize(data)
	
	data = merge_sort(data)
	result = verify_data(data)
	test_result("Merge Sort")
	
print "Success!"