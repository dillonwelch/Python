import argparse, os, re, shutil, sys

default_name = "NAME" # The default name in the skeleton project that needs to be replaced.
project_name = default_name # The name of the new project.
skeleton_name = "skeleton" # The name of the skeleton project.
starting_dir = os.getcwd()
exit_program_string = "Exiting the program."

#exe, project_name=default_name = sys.argv

""" Creates the new project folder and copies over all the files in the skeleton, replacing all instances of default_name with project_name """
def create_dir():
	# First copy the skeleton project, then change the current directory to the new project.
	shutil.copytree(skeleton_name, project_name)
	os.chdir(project_name)
	
	# Now walk through the new directory, replacing each occurence of default_name with project_name.
	for root, dirs, files in os.walk(os.getcwd()):
		# Go through the files.
		for f in files:
			os.rename(os.path.join(root, f), os.path.join(root, re.sub(default_name, project_name, f)))	# First rename the file.					
			try: # Now go through the file and replace all words within the file.
				with open(os.path.join(root, re.sub(default_name, project_name, f)), 'r+') as file: # Open the file for read + write.
					text = file.read() # Read in the file.
					file.seek(0) # Bug with Windows Python made this necessary.
					file.write(re.sub(default_name, project_name, text)) # Write the new file info.
			
			# Oh no, an exception!
			except IOError as e:
				print "Error with doing word replace within %r: %r" % (f, type(e))
				os.chdir(starting_dir)
				shutil.rmtree(project_name)
				sys.exit(1)
				
	# Separated because after renaming walk() couldn't find the new directory and so couldn't rename any files within.
	for root, dirs, files in os.walk(os.getcwd()):
		# Go through the directories and rename as necessary, the same way as with files.
		for d in dirs:
			old_name = d
			result = re.sub(default_name, project_name, d)
			os.rename(os.path.join(root, old_name), os.path.join(root, result))
		
""" Gets the new project name from the user. """
def get_dir_name():
	#print  # Get the project name.
	global project_name
	project_name = test_raw_input("Please input the new project name: ")
	
	test_dir_name(project_name)

def test_dir_name(dir_name):
	result = os.path.exists(dir_name) # Test the project name, if it exists ask if the user wants to overwrite.
	if result:
		overwrite_decision = False
		print "The directory already exists, would you like to overwrite it? Type y for yes or n for no."
		while not overwrite_decision:				
			overwrite = test_raw_input()
			if overwrite is "y":
				shutil.rmtree(os.path.join(os.getcwd(),dir_name))
				overwrite_decision = True
			elif overwrite is "n":
				get_dir_name()
				overwrite_decision = True
			else:
				print "Input not recognized. Type y for yes or n for no."
				
""" Gets raw input, and exits the program if exit is typed or EOF """
def test_raw_input(prompt=""):
	try:
		result = raw_input(prompt)
		list = ["EXIT"]
		if result.upper() in list:
			print exit_program_string
			sys.exit(0)
		else:
			return result
			
	except EOFError:
		print exit_program_string
		sys.exit(0)

		
""" Tests whether the skeleton directory exists, and exits the program if not. """		
def test_skeleton_dir():
	result = os.path.exists(skeleton_name)
	if not result:
		print "The skeleton directory could not be found at " + os.path.join(os.getcwd(), skeleton_name) + "."
		print exit_program_string
		sys.exit(1)
		
""" Tests input from the command line to see if there are any arguments """
def parse_test():
	parser = argparse.ArgumentParser(description='Test')
	parser.add_argument('-ProjectName', default=None, type=str, help='The name of the new project')
		
	global project_name
	project_name = parser.parse_args().ProjectName
	if project_name is not None:
		test_dir_name(project_name)
	else:
		get_dir_name()
	#print args.ProjectName

def main():
	parse_test()
	test_skeleton_dir()
	#get_dir_name()
	
	has_created = False
	# Create the new directory.
	while not has_created:
		try:
			create_dir()	
			print "The new project has been created with the name %s!" % project_name
			has_created = True
		
		except Exception as e:
			print "Fatal error: \n%r \n%s" % ({type(e), e.args, e}, exit_program_string)
			sys.exit(1)
main()
