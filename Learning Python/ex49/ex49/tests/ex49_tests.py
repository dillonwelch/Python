from nose.tools import *
import sys
sys.path.append("C:\Users\welchd\Desktop\My Stuff\Python\projects\ex48")
from ex48 import lexicon
from ex49 import parser

def setup():
    print "Hands up! This is a setup!"

def teardown():
    print "Did someone call for an exterminator?"

def test_basic():
	simple_sentence = parser.parse_sentence(lexicon.scan("go north"))
	assert_equal("player", simple_sentence.subject)
	assert_equal("go", simple_sentence.verb)
	assert_equal("north", simple_sentence.object)
	
def stop_test():
	stop_sentence = parser.parse_sentence(lexicon.scan("bear go the in of from at it north"))
	assert_equal("bear", stop_sentence.subject)
	assert_equal("go", stop_sentence.verb)
	assert_equal("north", stop_sentence.object)
	
@raises(parser.ParserError)
def empty_test_one():
	empty_sentence_one =  parser.parse_sentence("")
	
@raises(parser.ParserError)
def empty_test_two():
	empty_sentence_two =  parser.parse_sentence(lexicon.scan(""))
	
def error_tuple_test():
	error_tuple_setence = parser.parse_sentence(lexicon.scan("bear punch the wall"))