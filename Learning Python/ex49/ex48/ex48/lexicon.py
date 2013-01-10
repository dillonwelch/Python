direction_words = ('north', 'south', 'east', 'west', 'down', 'up', 'left', 'right', 'back')
verb_words = ('go', 'stop', 'kill', 'eat')
noun_words = ('door', 'bear', 'princess', 'cabinet')
stop_words = ('the', 'in', 'of', 'from', 'at', 'it')
direction_string = 'direction'
verb_string = 'verb'
stop_string = 'stop'
noun_string = 'noun'
lex = ((direction_string, direction_words), (verb_string, verb_words), (stop_string, stop_words), (noun_string, noun_words))
number_string = 'number'
error_string = 'error'

def convert_number(s):
	try:
		return int(s)
	except ValueError:
		return None

def scan(input):
	sentence = []
	lexicon = {}
	for name, words in lex:
		for word in words:
			lexicon[word] = name
	
	for word in input.split():
		temp = word.lower()
		word_type = lexicon.get(temp, error_string)
		if word_type == error_string:
			number = convert_number(temp)
			if number is not None:
				word_type = number_string
				word = number
				
		sentence.append((word_type, word))
	
	return sentence
	

	"""for word in direction_words:
		lexicon[word] = direction_string
	for word in verb_words:
		lexicon[word] = verb_string
	for word in noun_words:
		lexicon[word] = noun_string
	for word in stop_words:
		lexicon[word] = stop_string"""
	
"""
#lexicon = ((direction_string, direction_words), (verb_string, verb_words), (stop_string, stop_words), (noun_string, noun_words))
test = raw_input()
scan(test)

def scan(input):
	sentence = []
	for word in input.split():
		temp = word.lower()
		has_added = False
		for type in lexicon:
			name, words = type
			if any(temp in item for item in words):
				sentence.append((name, word))
				has_added = True
				break
				
		if not has_added:
			number = convert_number(temp)
			
			if number is not None:
				sentence.append((number_string, number))
			else:
				sentence.append((error_string, word))
				
	return sentence
"""